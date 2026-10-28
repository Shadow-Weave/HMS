"""Group A foundation tests: enums, exceptions, models, timeutil, text/json utils, configs, prompts, memory utils.
A 组基础测试：枚举、异常、模型、时间工具、文本/JSON 工具、配置、提示词、memory 工具。
"""

# Purpose (EN): Assert CONTRACT §2–§6 and §9 behaviour of the A1 files; A2 appends the schema/store tests (11–15).
# 目的（中）：断言 A1 文件对契约 §2–§6 与 §9 的行为；A2 追加模式/存储测试（11–15）。

# Input (EN): The A1 modules directly; tests 11-15 (schema/store) use the store/clock/ids fixtures of conftest.py.
# 输入（中）：直接使用 A1 模块；测试 11-15（模式/存储）使用 conftest.py 的 store/clock/ids 夹具。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside in-memory / tmp_path SQLite databases (no network, no home writes).
# 状态影响（中）：除内存 / tmp_path 的 SQLite 数据库外无副作用（无网络、不写入用户目录）。

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import json
import logging
import math
import pathlib
import re
import sqlite3
import threading
import time
import warnings
from datetime import date, datetime, timedelta, timezone

import pydantic
import pytest

from hms_elite import enums, exceptions, models, timeutil
from hms_elite.configs import base as cfg
from hms_elite.configs import prompts, registry
from hms_elite.index.base import INDEX_ATTR_KEYS
from hms_elite.memory import base as mbase
from hms_elite.memory import schema
from hms_elite.memory import storage as mstorage
from hms_elite.memory import utils as mutils
from hms_elite.utils import json as ujson
from hms_elite.utils import text as utext

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCOPE4 = {"user_id": "alice", "agent_id": "coder", "run_id": "run-billing"}
MEM0_CONFIG = {
    "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini", "temperature": 0.1, "top_p": 0.1, "top_k": 1}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}},
    "history_db_path": "/tmp/x/history.db",
    "version": "v1.1",
    "custom_instructions": None,
}
_ALLOWED_EDGES = {
    "A": set(),
    "B": {"A"},
    "C": {"A", "B"},
    "D": {"A", "B"},
    "E": {"A", "B", "C"},
    "F": {"A", "B", "D", "E"},
    "G": {"A", "B", "C", "D", "E", "F"},
    "H": {"A", "B", "C", "D", "E", "F", "G"},
}
_CJK = re.compile(r"[一-鿿]")
_HEADER_LINES = (
    "# Purpose (EN):",
    "# 目的（中）：",
    "# Input (EN):",
    "# 输入（中）：",
    "# Output (EN):",
    "# 输出（中）：",
    "# State effect (EN):",
    "# 状态影响（中）：",
)
_MARKERS = ("Input / 输入：", "Output / 输出：", "State effect / 状态影响：")


def _scope() -> models.Scope:
    """The S4 scope as a model.
    S4 范围的模型形式。
    """
    return models.Scope(**SCOPE4)


def _atom(**overrides) -> models.MemoryAtom:
    """A valid state atom with overrides applied.
    应用覆盖后的合法 state 原子。
    """
    base = dict(
        atom_id="atm_000001",
        memory_id="mem_000001",
        atom_type="state",
        content={"text": "User likes black coffee."},
        text="User likes black coffee.",
        scale_level="state",
        source_kind="observation",
        source_id="obs_000001",
        extracted_by="rule@obs-v1",
        scope=models.Scope(user_id="alice", run_id="run-coffee"),
        anchor_ids=["anc_000002"],
        obs_type="preference_event",
        created_at="2026-04-20T08:00:00Z",
        updated_at="2026-04-20T08:00:00Z",
        valid_from="2026-04-20T08:00:00Z",
    )
    base.update(overrides)
    return models.MemoryAtom(**base)


def _ep() -> models.EP:
    """The S4 EP1 as a model.
    S4 的 EP1 模型。
    """
    return models.EP(
        ep_id="ep_000001",
        slots=models.Slots(
            u="fix checkout webhook retry",
            b="Do not modify the subscription state machine.",
            a="patch src/billing/retry.ts: add exponential backoff",
            y="billing_e2e failed: subscription_state_regression",
            r="negative",
            kappa=models.Kappa(
                constraints=["Do not modify the subscription state machine."], evidence=["anc_000003", "anc_000002"]
            ),
        ),
        time_anchor="2026-06-15T10:04:00Z",
        anchor_ids=["anc_000003", "anc_000002"],
        scope=_scope(),
    )


def _item(item_id: str, slot: str) -> models.ContextItem:
    """A witnessed ContextItem in a slot.
    某槽位中一个带见证的 ContextItem。
    """
    return models.ContextItem(item_id=item_id, slot=slot, text="t", anchor_ids=["anc_000001"])


def _module_name(path: pathlib.Path) -> str:
    """Dotted module name of a repo file.
    仓库文件的点分模块名。
    """
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _group(module: str) -> str:
    """Owner group of an hms_elite module (CONTRACT §1).
    hms_elite 模块的归属组（契约 §1）。
    """
    # reranker/ and entities/ are provider packages like llms/embeddings/index (B → A only): they import
    # utils.factory, utils.text, models, enums and exceptions, never a plane (ingest/mvmi/graph/compiler/ops), so
    # neither adds a new allowed edge; llms/rule.py (B) importing entities/ (B) is an intra-group edge. telemetry.py
    # is the metering every provider wraps itself in (it imports only memory.utils), so it is B as well.
    # reranker/ 与 entities/ 是与 llms/embeddings/index 同类的提供者包（B → 仅 A）：它们只导入 utils.factory、
    # utils.text、models、enums 与 exceptions，从不导入任何平面（ingest/mvmi/graph/compiler/ops），因此都不新增
    # 允许边；llms/rule.py（B）导入 entities/（B）是组内边。telemetry.py 是每个提供者自行包裹的计量层（只导入
    # memory.utils），因此同属 B。
    if (
        module.startswith(
            (
                "hms_elite.llms",
                "hms_elite.embeddings",
                "hms_elite.index",
                "hms_elite.reranker",
                "hms_elite.entities",
            )
        )
        or module in {"hms_elite.utils.factory", "hms_elite.telemetry"}
    ):
        return "B"
    # segmentation.py is a pure text utility over utils.text (no store, index or graph): foundation group A, so the
    # ingest plane (C) may cut long messages with it and nothing forces it above A.
    # segmentation.py 是建立在 utils.text 之上的纯文本工具（无存储、索引或图）：基础组 A，因此摄取平面（C）可以
    # 用它切分长消息，而没有任何东西迫使它高于 A。
    if module == "hms_elite.segmentation":
        return "A"
    if module.startswith("hms_elite.ingest"):
        return "C"
    if module.startswith("hms_elite.mvmi") or module == "hms_elite.utils.scoring":
        return "D"
    if module.startswith("hms_elite.graph"):
        return "E"
    # compiler/coverage.py and compiler/scc.py are the step-8 coverage packer and its hierarchical refinement: F,
    # like compile.py that imports them (they reach B / D through llms.rule and mvmi.planner only).
    # compiler/coverage.py 与 compiler/scc.py 是第 8 步的覆盖式打包器及其分层下钻：与导入它们的 compile.py 同属 F
    # （它们只经 llms.rule 与 mvmi.planner 触及 B / D）。
    if module.startswith(("hms_elite.compiler.coverage", "hms_elite.compiler.scc")):
        return "F"
    if module.startswith("hms_elite.compiler"):
        return "F"
    if module.startswith("hms_elite.ops"):
        return "G"
    # hms4eye 兼容层位于门面之上（契约 §0.4 之外的新增子包），因此与门面同组
    if module.startswith("hms_elite.hms4eye"):  # the hms4eye layer sits above the facade, so it shares the facade group
        return "H"
    # The HTTP service face (the `server` extra) only wraps the facade, so it is group H like cli.py: it may import
    # every group and nothing imports it.
    # HTTP 服务面（`server` extra）只是门面的包装，因此与 cli.py 一样属于 H 组：它可以导入所有组，且没有模块导入它。
    if module.startswith("hms_elite.server"):
        return "H"
    # state/ (the §11 cards), eval/ (the ablation matrix), runtime.py (the agent adapter) and daemon.py (the
    # background loop) consume the facade — duck-typed, never imported — and the facade re-exports state/ and the
    # daemon, so all four are H: nothing below H may import them.
    # state/（§11 状态卡）、eval/（消融矩阵）、runtime.py（agent 适配器）与 daemon.py（后台循环）消费门面
    # ——鸭子类型，从不导入——而门面再导出 state/ 与守护进程，因此四者都是 H：H 以下的任何模块都不得导入它们。
    if module.startswith(("hms_elite.state", "hms_elite.eval")) or module in {"hms_elite.runtime", "hms_elite.daemon"}:
        return "H"
    # The async twin wraps the facade (it imports memory.main), so it shares the facade group H.
    # 异步孪生封装门面（导入 memory.main），因此与门面同属 H 组。
    if module in {
        "hms_elite",
        "hms_elite.memory.main",
        "hms_elite.memory.async_main",
        "hms_elite.migrate",
        "hms_elite.cli",
    }:
        return "H"
    return "A"


def _top_level_imports(tree: ast.Module) -> list[str]:
    """hms_elite modules imported at module level (including inside module-level try/except).
    模块级导入的 hms_elite 模块（含模块级 try/except 内的导入）。
    """
    found: list[str] = []

    def _visit(statements):
        for node in statements:
            if isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names if alias.name.startswith("hms_elite"))
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hms_elite"):
                found.append(node.module)
                found.extend(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Try):
                _visit(node.body)
                for handler in node.handlers:
                    _visit(handler.body)
                _visit(node.orelse)
                _visit(node.finalbody)
            elif isinstance(node, ast.If):
                _visit(node.body)
                _visit(node.orelse)

    _visit(tree.body)
    return found


def _resolve_module(name: str, known: set[str]) -> str | None:
    """Longest known module prefix of a dotted import target.
    点分导入目标的最长已知模块前缀。
    """
    parts = name.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return None


def _check_function_doc(qualname: str, func) -> None:
    """Assert the §0.3 bilingual layout of one public function docstring.
    断言某个公开函数文档字符串符合 §0.3 的双语布局。
    """
    doc = inspect.getdoc(func)
    assert doc, f"{qualname} lacks a docstring"
    lines = doc.split("\n")
    assert len(lines) >= 2, f"{qualname}: docstring needs an EN line followed by a ZH line"
    assert not _CJK.search(lines[0]), f"{qualname}: first docstring line must be English"
    assert _CJK.search(lines[1]), f"{qualname}: second docstring line must be Chinese"
    positions = [doc.find(marker) for marker in _MARKERS]
    assert all(p >= 0 for p in positions), f"{qualname}: missing Input/Output/State effect markers"
    assert positions == sorted(positions), f"{qualname}: markers out of order"
    input_block = doc[positions[0] : positions[1]].split("\n")[1:]
    for index, line in enumerate(input_block):
        match = re.match(r"^\s*([A-Za-z_]\w*): ", line)
        if match:
            following = input_block[index + 1] if index + 1 < len(input_block) else ""
            assert re.match(rf"^\s*{re.escape(match.group(1))}：", following), (
                f"{qualname}: argument {match.group(1)!r} needs a ZH line (full-width colon) after its EN line"
            )


def _public_functions(module):
    """Yield (qualname, function) for public functions/methods defined in a module.
    生成模块中定义的公开函数/方法的 (限定名, 函数)。
    """
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            yield f"{module.__name__}.{name}", obj
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for attr, member in vars(obj).items():
                if attr.startswith("_"):
                    continue
                func = member
                if isinstance(member, (classmethod, staticmethod)):
                    func = member.__func__
                elif isinstance(member, property):
                    func = member.fget
                if inspect.isfunction(func) and func.__module__ == module.__name__:
                    yield f"{module.__name__}.{name}.{attr}", func


def _load_module(path: pathlib.Path):
    """Import a repo file as a module (by dotted name, or from its path for examples/).
    以模块形式导入仓库文件（按点分名，examples/ 下按路径）。
    """
    name = _module_name(path)
    if name.startswith(("hms_elite", "tests")):
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_topology():
    """Module import graph respects §0.4: group edges, mvmi/scoring isolation, models never imports the compiler.
    模块导入图遵守 §0.4：组间边、mvmi/scoring 隔离、models 不导入编译器。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over the AST of every hms_elite file.
        对每个 hms_elite 文件 AST 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    files = sorted((ROOT / "hms_elite").rglob("*.py"))
    known = {_module_name(p) for p in files}
    assert "hms_elite.models" in known
    violations: list[str] = []
    for path in files:
        module = _module_name(path)
        group = _group(module)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target in _top_level_imports(tree):
            resolved = _resolve_module(target, known)
            if resolved is None or resolved == module:
                continue
            other = _group(resolved)
            if other != group and other not in _ALLOWED_EDGES[group]:
                violations.append(f"{module} ({group}) -> {resolved} ({other})")
            if module.startswith("hms_elite.mvmi") or module == "hms_elite.utils.scoring":
                assert not resolved.startswith(
                    ("hms_elite.graph", "hms_elite.compiler", "hms_elite.ops", "hms_elite.ingest")
                ), f"{module} must not import {resolved}"
            if module == "hms_elite.models":
                assert not resolved.startswith("hms_elite.compiler"), "models imports the compiler at module level"
            # The §2 tail modules receive the facade as an argument and never import it (no import cycle with H).
            # §2 尾部模块以参数形式接收门面，从不导入它（与 H 组之间没有导入环）。
            if module.startswith(("hms_elite.state", "hms_elite.eval")) or module in {
                "hms_elite.runtime",
                "hms_elite.daemon",
            }:
                assert not resolved.startswith(("hms_elite.memory.main", "hms_elite.memory.async_main")), (
                    f"{module} must not import the facade"
                )
            # segmentation and telemetry stay below every plane: A-only imports.
            # segmentation 与 telemetry 位于所有平面之下：只导入 A 组。
            if module in {"hms_elite.segmentation", "hms_elite.telemetry"}:
                assert _group(resolved) == "A", f"{module} must import group A only, not {resolved}"
    assert violations == [], "import topology violations: " + "; ".join(violations)


def test_enums_are_str_enums_and_values():
    """Every §2 enum is a (str, Enum) with the exact member set; derived constants match §2.
    §2 的每个枚举都是 (str, Enum) 且成员集精确；派生常量与 §2 一致。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over member values and constants.
        对成员值与常量的断言。

    State effect / 状态影响：
        None.
        无。
    """
    expected = {
        enums.EdgeType: ["temporal", "causal", "revision", "conflict", "use", "distill"],
        enums.FeedbackType: [
            "state_wrong", "state_outdated", "state_missing", "privacy_violation", "next_action_wrong",
            "relationship_not_useful", "connection_reason_wrong", "evidence_wrong", "render_not_like_me",
            "render_like_me", "object_important",
        ],
        enums.Scale: ["L1_micro", "L2_meso", "L3_macro", "state", "event", "entity", "relation"],
        enums.AtomType: ["message", "state", "event", "entity", "relation", "episode"],
        enums.ObservationType: ["fact", "preference_event", "habit", "constraint", "temporal_fact"],
        enums.Reward: ["positive", "negative", "neutral", "unknown"],
        enums.Visibility: ["private", "team", "event", "public"],
        enums.Role: ["user", "assistant", "system", "tool", "extracted"],
        enums.Sentiment: ["positive", "neutral", "negative", "mixed", "unknown"],
        enums.Slot: ["facts", "episodes", "preferences", "constraints", "skills", "audit"],
        enums.ObjectType: [
            "message", "tool_call", "file_diff", "test_log", "sensor_trace", "feedback", "checkpoint", "rollback",
            "system", "scheduler",
        ],
        enums.EventType: [
            "created", "updated", "deleted", "feedback", "checkpoint", "rollback", "tool_result", "scheduler",
        ],
        enums.NodeKind: ["message", "observation", "event", "entity", "ep", "interface_ep", "skill"],
        enums.SourceKind: [
            "message", "observation", "event", "entity", "ep", "interface_ep", "skill", "procedural", "feedback",
        ],
        enums.Decision: ["ADD", "REVISE", "CONFLICT", "NONE"],
        enums.HistoryEvent: [
            "ADD", "UPDATE", "DELETE", "ARCHIVE", "RESTORE", "FEEDBACK", "CONFLICT", "ROLLBACK", "PURGE",
        ],
        enums.Route: ["recall", "preference", "temporal", "entity", "work_state", "detective", "handoff"],
        enums.CompileMode: ["auto", "fast", "axis_graph", "deep"],
        enums.Relation: ["single_fact", "comparison", "habit", "supersession", "constraint"],
        enums.EntityType: ["PROPER", "IDENTIFIER", "FILE", "QUOTED", "TOPIC"],
        enums.MemoryType: ["semantic_memory", "episodic_memory", "procedural_memory"],
        enums.TransitionKind: ["ep", "interface_ep", "causal", "revision", "conflict", "rollback"],
        enums.InvariantKind: ["constraint", "failed_path", "stable_fact"],
        enums.ResidualKind: ["unresolved_conflict", "missing_outcome", "low_cohesion", "open_question"],
        enums.Phase: ["orient", "collect", "consolidate", "prune"],
        enums.ObligationName: ["evidence_coverage", "no_unresolved_conflict", "governance", "not_revoked"],
        enums.ObligationStatus: ["satisfied", "violated", "skipped"],
    }  # fmt: skip
    for cls, values in expected.items():
        assert issubclass(cls, str) and issubclass(cls, enums.StrEnum), cls
        assert cls.values() == values, cls
        for member in cls:
            if cls is not enums.MemoryType:
                assert member.name == member.value, member
            assert member == member.value and hash(member) == hash(member.value) and str(member) == member.value
    assert enums.FeedbackType.state_wrong == "state_wrong" and enums.EdgeType.causal == "causal"
    assert enums.MemoryType.PROCEDURAL == "procedural_memory"
    assert enums.EXTRACTABLE_OBJECT_TYPES == {"message", "tool_call", "file_diff", "test_log", "sensor_trace"}
    assert enums.INTERNAL_OBJECT_TYPES == {"feedback", "checkpoint", "rollback", "system", "scheduler"}
    assert enums.WHY_EDGE_TYPES == {"causal", "use", "distill", "revision", "conflict"}
    assert enums.VISIBILITY_RANK == {"private": 0, "team": 1, "event": 2, "public": 3}
    assert enums.rank(enums.Visibility.event) == 2
    assert enums.KIND_ORDER == {"ep": 0, "interface_ep": 1, "causal": 2, "revision": 3, "conflict": 4, "rollback": 5}
    assert enums.ABLATION_OF_MODE == {"fast": "axis", "axis_graph": "axis_graph", "deep": "axis_graph_support"}


def test_witnessed_models_reject_empty_anchor_ids():
    """Every witnessed model rejects anchor_ids == []; Slots enforces its closure.
    每个带见证的模型都拒绝 anchor_ids == []；Slots 强制闭包。

    Input / 输入：
        none.
        无。

    Output / 输出：
        pydantic.ValidationError for each empty witness list.
        每个空见证列表都触发 pydantic.ValidationError。

    State effect / 状态影响：
        None.
        无。
    """
    scope = _scope()
    ep = _ep()
    good_edge = dict(
        src_id="a", dst_id="b", edge_type="use", witness_anchor_ids=["anc_1"], created_by="ingest", scope=scope
    )
    good_iep = dict(
        interface_ep_id="iep_1",
        slots=ep.slots,
        slot_support={"u": ["n0"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"], "b": ["n0"], "kappa": ["n0", "n1"]},
        support_node_ids=["n0", "n1"],
        evidence_anchor_ids=["anc_1"],
        scope=scope,
    )
    good_tr = dict(
        source_kind="ep", source_id="ep_1", scale="L1_micro", s_t={}, a_t="a", s_t1={}, r_t="neutral", kappa_t={},
        w_t=["anc_1"], time_t="2026-06-15T10:04:00Z", time_t1="2026-06-15T10:04:00Z", scope=scope,
    )  # fmt: skip
    cases = [
        (
            models.Observation,
            dict(type="fact", predicate="is", text="t", date_observed="2026-01-01", scope=scope),
            "anchor_ids",
        ),
        (models.EP, ep.model_dump(), "anchor_ids"),
        (models.MemoryAtom, _atom().model_dump(), "anchor_ids"),
        (models.GraphNode, dict(node_id="n", node_kind="ep", ref_id="r", scale="L1_micro", scope=scope), "anchor_ids"),
        (models.GraphEdge, good_edge, "witness_anchor_ids"),
        (models.InterfaceEP, good_iep, "evidence_anchor_ids"),
        (models.ContextItem, dict(item_id="i", slot="facts", text="t"), "anchor_ids"),
        (models.Transition, good_tr, "w_t"),
    ]
    for cls, kwargs, field in cases:
        payload = {**kwargs, field: ["anc_1"]}
        # 非空形式可以正常构造
        cls(**payload)  # the non-empty form constructs
        with pytest.raises(pydantic.ValidationError):
            cls(**{**kwargs, field: []})
    with pytest.raises(pydantic.ValidationError, match="outcome or a constraint"):
        models.Slots(u="trigger", a="action")
    assert (
        models.Slots(u="trigger", a="action", kappa=models.Kappa(constraints=["c"])).gloss()
        == "[unknown] trigger → action → ?"
    )
    assert models.Invariant(text="t", support_node_ids=["n0"], anchor_ids=["anc_1"]).confidence == 0.7
    with pytest.raises(pydantic.ValidationError):
        models.Invariant(text="t", support_node_ids=["n0"], anchor_ids=[])
    assert models.Residual(text="t", kind="open_question").anchor_ids == []


def test_memory_atom_rejects_edge_keys_and_text_mismatch():
    """content with edge keys or content['text'] != text is rejected; hash and validity helpers work.
    含边键或 content['text'] != text 的 content 被拒绝；哈希与有效性辅助方法正常。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on MemoryAtom validators and helpers.
        对 MemoryAtom 校验器与辅助方法的断言。

    State effect / 状态影响：
        None.
        无。
    """
    for bad_content in ({"text": "User likes black coffee.", "edges": []}, {"text": "other"}, {"src_id": "x"}):
        with pytest.raises(pydantic.ValidationError):
            _atom(content=bad_content)
    atom = _atom(valid_to="2027-01-01T00:00:00Z")
    assert atom.hash == utext.md5_hex(atom.text)
    assert atom.expiration_date() == "2026-12-31"
    assert _atom().expiration_date() is None
    assert atom.is_valid_at("2026-05-01") and not atom.is_valid_at("2027-02-01") and not atom.is_valid_at("2026-01-01")
    assert not _atom(archived_reason="deleted").is_valid_at("2026-05-01")
    with pytest.raises(pydantic.ValidationError):
        _atom(valid_from="2026-05-01T00:00:00Z", valid_to="2026-04-01T00:00:00Z")
    assert "content_json" in models.RESERVED_ATOM_KEYS and "source_ep_ids_json" in models.RESERVED_ATOM_KEYS
    assert {"id", "memory", "primary_anchor_id", "expiration_date", "text_lemmatized"} <= models.RESERVED_ATOM_KEYS


def test_contextir_rejects_duplicate_item_ids_across_slots():
    """The same item_id in two slots (or a slot mismatch) is rejected; items()/anchor_ids() are deterministic.
    同一 item_id 出现在两个槽位（或槽位不匹配）被拒绝；items()/anchor_ids() 具有确定性。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the ContextIR validator.
        对 ContextIR 校验器的断言。

    State effect / 状态影响：
        None.
        无。
    """
    ok = models.ContextIR(
        query="q",
        as_of="2026-06-15T12:00:00Z",
        scope=_scope(),
        facts=[_item("a", "facts")],
        constraints=[_item("b", "constraints")],
    )
    assert [i.item_id for i in ok.items()] == ["a", "b"] and ok.anchor_ids() == ["anc_000001"]
    with pytest.raises(pydantic.ValidationError, match="appears in both"):
        models.ContextIR(
            query="q",
            as_of="2026-06-15T12:00:00Z",
            scope=_scope(),
            facts=[_item("a", "facts")],
            constraints=[_item("a", "constraints")],
        )
    with pytest.raises(pydantic.ValidationError, match="carries slot"):
        models.ContextIR(query="q", as_of="2026-06-15T12:00:00Z", scope=_scope(), facts=[_item("a", "audit")])


def test_timeutil_fixed_width_and_parsers():
    """to_utc_iso normalizes every accepted form to 32 chars; windows and helpers follow §9.1.
    to_utc_iso 将所有受支持形式归一化为 32 字符；窗口与辅助函数遵循 §9.1。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on timeutil.
        对 timeutil 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    inputs = [
        "2026-05-01T09:00:00Z",
        "2026-05-01T09:00:00+05:30",
        "2026-05-01T09:00:00",
        "2026-05-01",
        datetime(2026, 5, 1, 9, 0, tzinfo=timezone(timedelta(hours=2))),
        datetime(2026, 5, 1, 9, 0),
        date(2026, 5, 1),
        1_777_600_000,
        1_777_600_000.5,
        "2026-05-01T09:00:00.5+0000",
    ]
    for value in inputs:
        out = timeutil.to_utc_iso(value)
        assert len(out) == timeutil.ISO_WIDTH and out.endswith("+00:00"), value
    assert timeutil.to_utc_iso("2026-05-01T09:00:00+05:30") == "2026-05-01T03:30:00.000000+00:00"
    assert timeutil.to_utc_iso("2026-05-01") == "2026-05-01T00:00:00.000000+00:00"
    assert timeutil.to_utc_iso(0) == "1970-01-01T00:00:00.000000+00:00"
    assert len(timeutil.now_iso()) == 32
    # canonical strings take the fast path (returned as the same object) but their fields are still validated
    # 规范字符串走快速路径（返回同一对象），但其字段仍会被校验
    canonical = "2026-05-01T09:00:00.123456+00:00"
    assert timeutil.to_utc_iso(canonical) is canonical
    for shaped_garbage in ("2026-13-01T00:00:00.000000+00:00", "2026-05-01T25:00:00.000000+00:00"):
        with pytest.raises(exceptions.ValidationError) as info:
            timeutil.to_utc_iso(shaped_garbage)
        assert info.value.error_code == "VALIDATION_004"
    for garbage in ("nope", "", True, None, [1]):
        with pytest.raises(exceptions.ValidationError) as info:
            timeutil.to_utc_iso(garbage)
        assert info.value.error_code == "VALIDATION_004" and isinstance(info.value, ValueError)
    assert timeutil.parse_iso("2026-05-01T09:00:00Z").tzinfo is timezone.utc
    assert timeutil.date_of("2026-05-01T23:30:00-02:00") == "2026-05-02"
    assert timeutil.next_day_midnight("2026-12-31") == "2027-01-01T00:00:00.000000+00:00"
    assert timeutil.month_window(2026, 5) == ("2026-05-01T00:00:00.000000+00:00", "2026-05-31T23:59:59.999999+00:00")
    assert timeutil.month_window(2026, 12)[1] == "2026-12-31T23:59:59.999999+00:00"
    assert timeutil.week_window("2026-09-02") == (
        "2026-08-31T00:00:00.000000+00:00",
        "2026-09-06T23:59:59.999999+00:00",
    )
    assert timeutil.day_window("2026-05-06") == ("2026-05-06T00:00:00.000000+00:00", "2026-05-06T23:59:59.999999+00:00")
    assert timeutil.year_window(2025) == ("2025-01-01T00:00:00.000000+00:00", "2025-12-31T23:59:59.999999+00:00")
    assert timeutil.age_days("2026-01-01", "2026-01-03T12:00:00Z") == 2.5
    assert timeutil.age_days("2026-01-03", "2026-01-01") == 0.0
    assert timeutil.shift_days("2026-02-28", 1) == "2026-03-01"


def test_text_tokenize_entities_topics_dates():
    """tokenize keeps identifiers; entities/topics/dates follow §9.2 (May rule, ZH month, PROPER stop-list).
    tokenize 保留标识符；实体/主题/日期遵循 §9.2（May 规则、中文月份、PROPER 停用表）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on utils.text.
        对 utils.text 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert utext.tokenize("patch retry.ts and billing_e2e") == ["patch", "retry.ts", "and", "billing_e2e"]
    assert utext.tokenize("我喜欢") == ["我喜", "喜欢"] and utext.tokenize("茶") == ["茶"]
    assert utext.identifier_tokens("patch src/billing/retry.ts and billing_e2e, retry.ts again") == [
        "src/billing/retry.ts",
        "billing_e2e",
        "retry.ts",
    ]
    assert utext.is_identifier("retry.ts") and utext.is_identifier("oat-milk") and not utext.is_identifier("coffee")
    stems = [utext.stem(t) for t in ["likes", "lattes", "switched", "testing", "timing", "cities", "was", "retry.ts"]]
    assert stems == ["like", "latte", "switch", "test", "timing", "city", "was", "retry.ts"]
    assert utext.content_tokens("User likes oat-milk lattes in the morning.") == ["like", "oat-milk", "latte", "morn"]
    sarah = utext.extract_entities("Sarah joined the billing team in March.")
    assert [(c.entity_type, c.text, c.key) for c in sarah] == [("PROPER", "Sarah", "sarah")]
    s4 = utext.extract_entities(
        "file_diff src/billing/retry.ts: add exponential backoff (intent: fix checkout webhook retry)"
    )
    assert [(c.entity_type, c.key, c.normalized) for c in s4] == [
        ("IDENTIFIER", "file_diff", "file_diff"),
        ("FILE", "retry.ts", "src/billing/retry.ts"),
    ]
    quoted = utext.extract_entities("'patch src/billing/retry.ts: add exponential backoff' failed")
    assert [c.key for c in quoted] == ["retry.ts"]
    multi = utext.extract_entities("Yesterday I visited the Berlin office with Sarah Connor in New York.")
    assert [c.text for c in multi] == ["Berlin", "Sarah Connor", "New York"]
    assert utext.extract_entities("Do not modify the subscription state machine.") == []
    assert (
        utext.entity_key("src/billing/retry.ts") == "retry.ts"
        and utext.entity_key("  Sarah  Connor ") == "sarah connor"
    )
    assert utext.normalize_entity("  New   York ") == "new york"
    assert utext.topic_tags("patched retry.ts to add backoff billing_e2e failed the webhook kept timing out") == [
        "engineering",
        "billing",
    ]
    assert utext.topic_tags("I like oat-milk lattes in the morning.") == ["coffee"]
    assert utext.topic_tags("Sarah joined the billing team in March.") == ["work", "billing"]
    assert utext.topic_tags("我喜欢咖啡") == ["coffee"] and utext.topic_tags("nothing here") == []
    may = utext.extract_dates("what happened in May", base_date="2026-09-02")
    assert may == [("month_en", "2026-05-01T00:00:00.000000+00:00", "2026-05-31T23:59:59.999999+00:00")]
    assert utext.extract_dates("retry.ts may change", base_date="2026-09-02") == []
    assert utext.extract_dates("5月发生了什么", base_date="2026-09-02") == [
        ("month_zh", "2026-05-01T00:00:00.000000+00:00", "2026-05-31T23:59:59.999999+00:00")
    ]
    assert utext.resolve_date("Sarah joined the billing team in March.", base_date="2026-03-02") == "2026-03-01"
    assert (
        utext.resolve_date("Sarah shipped the webhook retry design on May 6.", base_date="2026-05-06") == "2026-05-06"
    )
    assert (
        utext.extract_dates("Sarah is reviewing the checkout refactor this week.", base_date="2026-06-11")[0][0]
        == "relative_this"
    )
    assert utext.resolve_date("Yesterday I visited the Berlin office.", base_date="2026-05-10") == "2026-05-09"
    assert utext.resolve_date("Marcus decided to separate the mayor's office.", base_date="2026-05-10") is None
    kinds = [
        w[0]
        for w in utext.extract_dates("3 days ago, in 2 weeks, 2026-05, 2025-12-24 and 2周前", base_date="2026-09-02")
    ]
    assert kinds == ["ago", "in_n", "iso_month", "iso_date", "ago"]
    assert utext.extract_dates("last month", base_date="2026-01-15") == [("relative", *timeutil.month_window(2025, 12))]
    assert utext.sentences("Hello there. How are you?\nFine thanks") == ["Hello there.", "How are you?", "Fine thanks"]
    assert utext.ordered_union(["b", "a"], ["a", "c"]) == ["b", "a", "c"]


def test_text_rewrite_and_closure_helpers():
    """rewrite_first_person, clean_object, sentiment, is_first_person and the action/cause regexes follow §9.2/§11.2.
    rewrite_first_person、clean_object、sentiment、is_first_person 与动作/原因正则遵循 §9.2/§11.2。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on utils.text helpers.
        对 utils.text 辅助函数的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert utext.rewrite_first_person("I patched retry.ts to add backoff") == "User patched retry.ts to add backoff."
    assert (
        utext.rewrite_first_person("Yesterday I visited the Berlin office.")
        == "Yesterday User visited the Berlin office."
    )
    assert utext.rewrite_first_person("I've done it and my team loves it; we ship daily") == (
        "User has done it and user's team loves it; the team ship daily."
    )
    assert utext.rewrite_first_person("I'm happy!") == "User is happy!"
    assert utext.clean_object("black coffee now") == "black coffee"
    assert utext.clean_object("the Berlin office") == "Berlin office"
    assert utext.clean_object("oat-milk lattes in the morning") == "oat-milk lattes in the morning"
    assert utext.clean_object("黑咖啡了") == "黑咖啡"
    assert utext.sentiment("billing_e2e failed") == "negative"
    assert utext.sentiment("I like it but it failed") == "mixed"
    assert utext.sentiment("Sarah joined the team") == "neutral"
    assert utext.sentiment("tests passed") == "positive"
    assert utext.is_first_person("Yesterday I visited the Berlin office.")
    assert utext.is_first_person("Actually I switched to black coffee now.")
    assert utext.is_first_person("我喜欢燕麦拿铁。") and not utext.is_first_person("Sarah joined the team.")
    sentence = "I patched retry.ts to add backoff because the webhook kept timing out, and billing_e2e failed."
    match = utext.ACTION_RE.search(sentence)
    assert match["verb"] == "patched" and match["target"] == "retry.ts to add backoff"
    assert utext.CAUSE_RE.search(sentence)["u"] == "the webhook kept timing out"
    assert utext.CLAUSE_SPLIT_RE.split(sentence.rstrip("."))[-1] == "billing_e2e failed"
    assert utext.has_action_closure(sentence) and not utext.has_action_closure("I like coffee.")
    assert utext.has_action_closure("We patched retry.ts. Do not modify the state machine.")
    assert (
        utext.CONSTRAINT_RE.match("Do not modify the subscription state machine")["rest"]
        == "modify the subscription state machine"
    )
    assert utext.sha256_hex("a") == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    assert utext.md5_hex("a") == "0cc175b9c0f1b6a831c399e269772661"


def test_parse_llm_json_ladder(caplog):
    """Fenced, prose-wrapped and top-level-array replies parse; garbage yields [] plus a logged error.
    围栏、散文包裹与顶层数组回复均可解析；垃圾输入返回 [] 并记录错误日志。

    Input / 输入：
        caplog: pytest log capture fixture.
        caplog：pytest 日志捕获夹具。

    Output / 输出：
        Assertions on utils.json.
        对 utils.json 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert ujson.parse_llm_json('```json\n{"observations": [{"id": "0"}]}\n```', key="observations") == [{"id": "0"}]
    assert ujson.parse_llm_json('Sure! Here you go: {"eps": []} — done.', key="eps") == []
    assert ujson.parse_llm_json('Sure! Here you go: {"eps": [1]} — done.', key="eps") == [1]
    assert ujson.parse_llm_json('list: [{"a": 1}] end', key=None) == [{"a": 1}]
    assert ujson.parse_llm_json('<think>hmm</think>{"decisions": [1, 2]}', key="decisions") == [1, 2]
    assert ujson.parse_llm_json("", key="x") == [] and ujson.parse_llm_json("   ", key="x") == []
    assert ujson.parse_llm_json('{"interface_ep": {"slots": {}}}', key="interface_ep") == {"slots": {}}
    assert ujson.parse_llm_json('{"interface_ep": null}', key="interface_ep") == []
    assert ujson.parse_llm_json('{"observations": "not a list"}', key="observations") == []
    assert ujson.parse_llm_json('{"a": 1}', key=None) == {"a": 1}
    assert ujson.parse_llm_json("[1, 2]", key="observations") == []
    with caplog.at_level(logging.ERROR, logger="hms_elite.utils.json"):
        assert ujson.parse_llm_json("total garbage {{{", key="observations", task="observation_extraction") == []
    assert any("Error parsing observation_extraction response" in rec.getMessage() for rec in caplog.records)
    assert ujson.remove_code_blocks('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert ujson.remove_code_blocks(["a", "b"]) == "a\nb"
    assert ujson.extract_json('x {"a": [1]} y') == '{"a": [1]}' and ujson.extract_json("plain") == "plain"
    assert ujson.canonical_json({"b": 1, "a": "中"}) == '{"a":"中","b":1}'


def test_prompt_builders_round_trip():
    """parse_sections inverts every builder; prompts carry HMS_TASK markers and exact §6.3 OUTPUT FORMAT blocks.
    parse_sections 是每个构造器的逆；提示词带 HMS_TASK 标记与精确的 §6.3 OUTPUT FORMAT 块。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on configs.prompts.
        对 configs.prompts 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert prompts.TASKS == (
        "observation_extraction",
        "observation_extraction_compact",
        "ep_extraction",
        "revision_decision",
        "edge_proposal",
        "coarsen",
        "procedural_summary",
    )
    assert set(prompts.PROMPTS) == set(prompts.TASKS) and prompts.PAST_MESSAGE_TRUNCATION_LIMIT == 300
    assert set(prompts.ROOT_KEY) == set(prompts.TASKS) - {"procedural_summary"}
    # The composite task is the default add() round trip: its own table, disjoint from the staged tasks and from
    # ROOT_KEY (generate_json hands the whole multi-root object back), same header contract as every prompt.
    # 复合任务是默认的 add() 往返：自成一表，与分段任务及 ROOT_KEY 都不相交（generate_json 交回整个多根对象），
    # 头部契约与每个提示词相同。
    assert prompts.COMPOSITE_TASKS == ("memory_extraction", "memory_extraction_compact", "memory_extraction_lines")
    assert set(prompts.COMPOSITE_PROMPTS) == set(prompts.COMPOSITE_TASKS)
    assert not set(prompts.COMPOSITE_TASKS) & set(prompts.TASKS) and not set(prompts.COMPOSITE_TASKS) & set(prompts.ROOT_KEY)
    assert prompts.MEMORY_EXTRACTION_ROOTS == ("observations", "eps", "decisions")
    for task, prompt in prompts.COMPOSITE_PROMPTS.items():
        assert prompt.split("\n")[0] == f"HMS_TASK: {task}" and prompts.task_of(prompt) == task
        for heading in ("# ROLE", "# INPUTS", "# GUIDELINES", "# EXAMPLES", "# OUTPUT FORMAT"):
            assert heading in prompt, (task, heading)
        section = prompts.output_format_section(prompt)
        if task == "memory_extraction_lines":
            # The lines twin (§6.6) answers in tab-separated text: its OUTPUT FORMAT names the three sections and
            # carries no JSON at all.
            # 行式孪生（§6.6）以 tab 分隔文本回答：其 OUTPUT FORMAT 点名三个分节，完全不含 JSON。
            assert all(f"\n{name}\n" in section for name in ("OBS", "EP", "DEC")) and "{" not in section, task
            continue
        assert set(json.loads(ujson.extract_json(section))) == set(prompts.MEMORY_EXTRACTION_ROOTS), task
    for task, prompt in prompts.PROMPTS.items():
        assert prompt.split("\n")[0] == f"HMS_TASK: {task}" and prompts.task_of(prompt) == task
        for heading in ("# ROLE", "# INPUTS", "# GUIDELINES", "# EXAMPLES", "# OUTPUT FORMAT"):
            assert heading in prompt, (task, heading)
        section = prompts.output_format_section(prompt)
        if task in prompts.ROOT_KEY:
            obj = json.loads(ujson.extract_json(section))
            assert list(obj) == [prompts.ROOT_KEY[task]], task
            assert section.endswith(prompts.JSON_TRAILER)
        else:
            positions = [section.find(h) for h in prompts.PROCEDURAL_SUMMARY_HEADINGS]
            assert all(p >= 0 for p in positions) and positions == sorted(positions)
            assert section.endswith("Return ONLY the markdown above. No JSON, no wrappers.")
    assert prompts.task_of("no marker here") is None
    existing = [{"id": "0", "text": "User likes oat-milk lattes in the morning."}]
    new_messages = [
        {
            "idx": 0,
            "role": "user",
            "text": "Actually I switched to black coffee now.",
            "timestamp": "2026-04-20T08:00:00+00:00",
        }
    ]
    built = prompts.build_observation_prompt(
        existing_memories=existing,
        new_messages=new_messages,
        last_messages=[{"role": "user", "content": "a\nb"}, {"role": "assistant", "content": "x" * 400}],
        observation_date="2026-04-20",
        current_date="2026-09-02",
        custom_instructions="Be terse.",
    )
    sections = prompts.parse_sections(built)
    assert built.endswith("# Output:")
    assert list(sections) == [
        "Last k Messages",
        "Existing Memories",
        "New Messages",
        "Observation Date",
        "Current Date",
        "Custom Instructions",
    ]
    assert sections["Existing Memories"] == existing and sections["New Messages"] == new_messages
    assert sections["Observation Date"] == "2026-04-20" and sections["Current Date"] == "2026-09-02"
    assert sections["Custom Instructions"] == "Be terse."
    assert sections["Last k Messages"] == "user: a b\nassistant: " + "x" * 300
    no_custom = prompts.parse_sections(
        prompts.build_observation_prompt(
            existing_memories=[],
            new_messages=[],
            last_messages=[],
            observation_date="2026-04-20",
            current_date="2026-09-02",
        )
    )
    assert (
        "Custom Instructions" not in no_custom
        and no_custom["Existing Memories"] == []
        and no_custom["Last k Messages"] == ""
    )
    events = [
        {"idx": 0, "role": "assistant", "object_type": "message", "text": "t", "timestamp": "2026-06-15T10:04:00+00:00"}
    ]
    constraints = [{"id": "c0", "text": "Do not modify the subscription state machine."}]
    recent = [{"id": "e0", "u": "u", "a": "a", "y": "y", "r": "negative"}]
    ep_sections = prompts.parse_sections(
        prompts.build_ep_prompt(
            events=events, known_constraints=constraints, recent_eps=recent, observation_date="2026-06-15"
        )
    )
    assert ep_sections == {
        "Events": events,
        "Known Constraints": constraints,
        "Recent EPs": recent,
        "Observation Date": "2026-06-15",
    }
    new_items = [
        {
            "id": "0",
            "type": "preference_event",
            "subject": "user",
            "predicate": "likes",
            "objects": ["black coffee"],
            "relation": "supersession",
            "text": "User likes black coffee.",
            "date_observed": "2026-04-20",
        }
    ]
    candidates = [{**new_items[0], "id": "2", "valid_from": "2026-04-02T08:00:00+00:00"}]
    rev = prompts.parse_sections(
        prompts.build_revision_prompt(new_items=new_items, candidates=candidates, current_date="2026-09-02")
    )
    assert rev == {"New Memories": new_items, "Candidate Existing Memories": candidates, "Current Date": "2026-09-02"}
    node = {
        "ref": "n0",
        "kind": "ep",
        "u": "u",
        "b": "",
        "a": "a",
        "y": "y",
        "r": "negative",
        "subject": None,
        "predicate": None,
        "objects": [],
        "time": "2026-06-15T10:04:00+00:00",
        "entities": ["retry.ts"],
        "topics": ["engineering"],
        "anchor_refs": ["w0"],
    }
    witnesses = [{"ref": "w0", "node_ref": "n0", "source_ref": "diff://retry.ts", "time": "2026-06-15T10:04:00+00:00"}]
    edge_sections = prompts.parse_sections(
        prompts.build_edge_prompt(
            nodes=[node, {"ref": "n1", "kind": "observation", "a": "text"}], pairs=[["n0", "n1"]], witnesses=witnesses
        )
    )
    assert (
        edge_sections["Nodes"][0] == node
        and edge_sections["Candidate Pairs"] == [["n0", "n1"]]
        and edge_sections["Witnesses"] == witnesses
    )
    for key in ("u", "b", "a", "y", "r", "subject", "predicate", "objects"):
        assert key in edge_sections["Nodes"][1], key
    assert edge_sections["Nodes"][1]["r"] == "unknown" and edge_sections["Nodes"][1]["objects"] == []
    lines = edge_sections["Allowed Edge Types"].split("\n")
    assert [line.split(":")[0] for line in lines] == ["temporal", "causal", "revision", "conflict", "use", "distill"]
    assert "u, b, a, y, r" in prompts.EDGE_PROPOSAL_PROMPT and "anchor_refs" in prompts.EDGE_PROPOSAL_PROMPT
    support = [
        {
            "ref": "n0",
            "kind": "ep",
            "u": "u",
            "b": "",
            "a": "a",
            "y": "y",
            "r": "negative",
            "constraints": [],
            "time": "t",
            "anchor_refs": ["w0"],
        }
    ]
    internal = [{"ref": "x0", "src": "n0", "dst": "n1", "type": "causal"}]
    boundary = [{"ref": "b0", "inner": "n1", "outer": "outside", "direction": "out", "type": "use"}]
    co = prompts.parse_sections(
        prompts.build_coarsen_prompt(
            support_nodes=support, internal_edges=internal, boundary_edges=boundary, target_scale="L2_meso"
        )
    )
    assert co == {
        "Support Nodes": support,
        "Internal Edges": internal,
        "Boundary Edges": boundary,
        "Target Scale": "L2_meso",
    }
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}]},
        {"role": "tool", "content": None},
        {"role": "assistant", "content": "ok"},
    ]
    procedural = prompts.build_procedural_messages(messages, None)
    assert procedural[0] == {"role": "system", "content": prompts.PROCEDURAL_SUMMARY_PROMPT}
    assert procedural[1:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Create procedural memory of the above conversation."},
    ]
    assert prompts.build_procedural_messages([], "custom")[0]["content"] == "custom"


def test_config_mem0_dict_and_negative_cases(monkeypatch, tmp_path):
    """The stock Mem0 dict fails ONLY with CONFIG_002 on qdrant; renamed/removed/unknown keys and literals follow §5.
    原生 Mem0 字典仅因 qdrant 触发 CONFIG_002；改名/移除/未知键与字面量行为遵循 §5。

    Input / 输入：
        monkeypatch: pytest monkeypatch fixture (sets HMS_DIR).
        monkeypatch：pytest monkeypatch 夹具（设置 HMS_DIR）。

        tmp_path: pytest temporary directory.
        tmp_path：pytest 临时目录。

    Output / 输出：
        Assertions on HMSConfig.from_dict and friends.
        对 HMSConfig.from_dict 及相关函数的断言。

    State effect / 状态影响：
        None.
        无。
    """
    monkeypatch.setenv("HMS_DIR", str(tmp_path))
    fixture = copy.deepcopy(MEM0_CONFIG)
    with pytest.raises(exceptions.ConfigError) as info:
        cfg.HMSConfig.from_dict(fixture)
    assert info.value.error_code == "CONFIG_002" and info.value.details == {
        "kind": "index",
        "provider": "qdrant",
        "supported": ["memory", "milvus", "sqlite"],
    }
    assert info.value.suggestion == (
        "HMS Elite ships 'sqlite' and 'memory' index providers ('milvus' via the optional [milvus] extra); "
        "use {'provider': 'sqlite'}"
    )
    assert fixture == MEM0_CONFIG
    ok = copy.deepcopy(MEM0_CONFIG)
    ok["vector_store"]["provider"] = "sqlite"
    config = cfg.HMSConfig.from_dict(ok)
    assert config.store.path == ok["history_db_path"] == config.db_path() and config.version == "v1.1"
    assert config.llm.provider == "openai" and config.llm.config["top_k"] == 1 and config.embedder.provider == "openai"
    with pytest.warns(DeprecationWarning):
        renamed = cfg.HMSConfig.from_dict(
            {"index": {"provider": "memory"}, "db_path": "/tmp/y.db", "custom_fact_extraction_prompt": "p"}
        )
    assert (
        renamed.vector_store.provider == "memory"
        and renamed.store.path == "/tmp/y.db"
        and renamed.custom_instructions == "p"
    )
    for removed in ("graph_store", "enable_graph"):
        with pytest.raises(exceptions.ConfigError) as info:
            cfg.HMSConfig.from_dict({removed: True})
        assert info.value.error_code == "CONFIG_003" and info.value.details["key"] == removed
    with pytest.raises(exceptions.ConfigError) as info:
        cfg.HMSConfig.from_dict({"scoreing": {}})
    assert info.value.error_code == "CONFIG_001" and "scoreing" in str(info.value) and "scoring" in str(info.value)
    with pytest.raises(exceptions.ConfigError) as info:
        cfg.HMSConfig.from_dict({"store": {"pth": "x"}})
    assert info.value.error_code == "CONFIG_001" and "'pth'" in str(info.value) and "'path'" in str(info.value)
    for envelope in ({"provider": "bogus"}, {"provider": "bogus", "config": {"x": 1}}):
        for key in ("llm", "embedder", "vector_store"):
            with pytest.raises(exceptions.ConfigError) as info:
                cfg.HMSConfig.from_dict({key: envelope})
            assert info.value.error_code == "CONFIG_002" and info.value.details["provider"] == "bogus"
    assert cfg.HMSConfig.from_dict({"llm": {"provider": "rule", "config": None}}).llm.config == {}
    defaults = cfg.HMSConfig()
    assert (defaults.llm.provider, defaults.embedder.provider, defaults.vector_store.provider) == (
        "rule",
        "hash",
        "sqlite",
    )
    assert defaults.store.path == str(tmp_path / "hms.db") and defaults.compiler.default_mode == "auto"
    assert (
        defaults.scheduler.phases == ["orient", "collect", "consolidate", "prune"]
        and defaults.visibility_default == "private"
    )
    assert (
        defaults.ingest.emit_relation_atoms is True
        and defaults.graph.coarsen_l3 is True
        and defaults.scheduler.decay_rate == 0.1
    )
    # Retention is an escape hatch: every switch is off by default, the threshold is 90 days, negatives are rejected.
    # 保留策略是逃生舱：所有开关默认关闭，阈值 90 天，负数被拒绝。
    assert (defaults.retention.enabled, defaults.retention.run_in_prune, defaults.retention.older_than_days) == (
        False,
        False,
        90,
    )
    bad_literals = [
        {"version": "v9"},
        {"store": {"journal_mode": "x"}},
        {"ingest": {"revision_mode": "x"}},
        {"graph": {"coarsen_policy": "x"}},
        {"graph": {"causal_proposer": "x"}},
        {"compiler": {"default_mode": "x"}},
        {"scheduler": {"phases": ["x"]}},
        {"retention": {"older_than_days": -1}},
        {"visibility_default": "x"},
    ]
    for bad in bad_literals:
        with pytest.raises(exceptions.ConfigError) as info:
            cfg.HMSConfig.from_dict(bad)
        assert info.value.error_code == "CONFIG_001", bad
    with pytest.raises(exceptions.ConfigError) as info:
        cfg.HMSConfig.from_dict(["not", "a", "dict"])
    assert info.value.error_code == "CONFIG_001"
    # the reranker envelope is a registered kind: unknown providers are CONFIG_002, known ones build silently
    # reranker 信封是已注册的种类：未知提供者为 CONFIG_002，已知提供者静默构造
    with pytest.raises(exceptions.ConfigError) as info:
        cfg.HMSConfig.from_dict({"reranker": {"provider": "cohere", "config": None}})
    assert info.value.error_code == "CONFIG_002" and info.value.details == {
        "kind": "reranker",
        "provider": "cohere",
        "supported": ["llm", "overlap"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        overlap = cfg.HMSConfig.from_dict({"reranker": {"provider": "overlap", "config": None}})
    assert overlap.reranker is not None and (overlap.reranker.provider, overlap.reranker.config) == ("overlap", {})
    assert cfg.HMSConfig().reranker is None and cfg.RerankerEnvelope().provider == "overlap"
    assert registry.supported("reranker") == ["llm", "overlap"]
    # Every registry kind lists its providers sorted, the later providers (deepseek / ollama / fastembed / ark /
    # spacy / milvus) included, and the index package re-exports the Milvus provider beside sqlite / memory.
    # 每种注册表都按排序列出其提供者，含后加入的提供者（deepseek / ollama / fastembed / ark / spacy / milvus），
    # 索引包也在 sqlite / memory 之外再导出 Milvus 提供者。
    assert registry.supported("llm") == ["anthropic", "deepseek", "ollama", "openai", "rule", "scripted"]
    assert registry.supported("embedder") == ["ark", "fastembed", "hash", "ollama", "openai"]
    assert registry.supported("entity_extractor") == ["rule", "spacy"]
    assert all(registry.supported(kind) == sorted(registry.supported(kind)) for kind in registry.REGISTRY)
    from hms_elite import index as index_pkg

    assert {"MilvusIndex", "MilvusIndexConfig", "SQLiteIndex", "MemoryIndex"} <= set(index_pkg.__all__)
    assert index_pkg.MilvusIndex.__name__ == "MilvusIndex"
    with pytest.warns(UserWarning, match="history_db_path"):
        both = cfg.HMSConfig.from_dict({"history_db_path": "/tmp/a.db", "store": {"path": "/tmp/b.db"}})
    assert both.store.path == "/tmp/a.db"
    with pytest.raises(pydantic.ValidationError):
        cfg.HMSConfig(vector_store={"provider": "qdrant"})
    with pytest.raises(pydantic.ValidationError):
        cfg.HMSConfig(store=cfg.StoreConfig(), bogus=1)
    with pytest.raises(pydantic.ValidationError):
        cfg.HMSConfig().version = "v2"
    assert registry.supported("index") == ["memory", "milvus", "sqlite"] and registry.EXTRA_OF_PROVIDER == {
        "openai": "openai",
        "anthropic": "anthropic",
        "milvus": "milvus",
        "deepseek": "openai",
        "ollama": "ollama",
        "fastembed": "fastembed",
        "spacy": "spacy",
        "ark": "ark",
    }
    # Every extra hint names a real [project.optional-dependencies] group, so a DEPS_001 suggestion is installable.
    # 每个 extra 提示都指向真实的 [project.optional-dependencies] 分组，因此 DEPS_001 的建议是可安装的。
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(pyproject["project"]["optional-dependencies"])
    assert set(registry.EXTRA_OF_PROVIDER.values()) <= extras
    assert registry.REGISTRY["llm"]["rule"] == ("hms_elite.llms.rule.RuleLLM", "hms_elite.llms.rule.RuleLlmConfig")
    source = copy.deepcopy(MEM0_CONFIG)
    normalized = cfg.normalize_mem0_keys(source, strict=False)
    assert normalized == MEM0_CONFIG and normalized is not source
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        twice = cfg.normalize_mem0_keys(
            cfg.normalize_mem0_keys({"index": {"provider": "memory"}}, strict=False), strict=False
        )
    assert twice == {"vector_store": {"provider": "memory"}} and len(caught) == 1


def test_exceptions_codes_and_unwrap():
    """Every exception carries its default code; ValidationError is a ValueError; pydantic wrappers unwrap.
    每个异常携带默认错误码；ValidationError 是 ValueError；pydantic 包装可被解包。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on hms_elite.exceptions.
        对 hms_elite.exceptions 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    expected = {
        exceptions.HMSError: "HMS_000",
        exceptions.ValidationError: "VALIDATION_000",
        exceptions.ConfigError: "CONFIG_001",
        exceptions.LLMError: "LLM_001",
        exceptions.EmbeddingError: "EMBED_001",
        exceptions.StoreError: "STORE_001",
        exceptions.WitnessMissingError: "WITNESS_001",
        exceptions.ProofObligationError: "PCIR_001",
        exceptions.DependencyError: "DEPS_001",
    }
    for cls, code in expected.items():
        err = cls("boom")
        assert err.error_code == code and err.message == "boom" and str(err) == "boom" and err.details == {}
        assert repr(err) == f"{cls.__name__}(message='boom', error_code={code!r})"
        assert err.to_dict()["error_type"] == cls.__name__ and err.to_dict()["error_code"] == code
    assert isinstance(exceptions.ValidationError("x"), ValueError) and isinstance(
        exceptions.ConfigError("x"), ValueError
    )
    assert issubclass(exceptions.WitnessMissingError, exceptions.StoreError)
    assert exceptions.LLMError("x").suggestion == "Please check your LLM configuration and API key"
    assert exceptions.WitnessMissingError("x").suggestion.startswith(
        "Every HMS object must reference at least one anchor_id"
    )
    dep = exceptions.DependencyError("no openai", details={"package": "openai", "extra": "openai"})
    assert dep.suggestion == "pip install 'hms-elite[openai]'"
    custom = exceptions.ValidationError(
        "bad", "VALIDATION_009", details={"top_k": -1}, suggestion="fix it", debug_info={"k": 1}
    )
    assert custom.to_dict() == {
        "error_type": "ValidationError",
        "error_code": "VALIDATION_009",
        "message": "bad",
        "details": {"top_k": -1},
        "suggestion": "fix it",
        "debug_info": {"k": 1},
    }
    with pytest.raises(pydantic.ValidationError) as info:
        models.Scope(user_id="  ")
    inner = exceptions.unwrap_hms_error(info.value)
    assert isinstance(inner, exceptions.ValidationError) and inner.error_code == "VALIDATION_001"
    assert exceptions.unwrap_hms_error(ValueError("plain")) is None and exceptions.unwrap_hms_error(custom) is custom


def test_scope_key_and_matches():
    """Scope trims ids, builds the Mem0 scope key with escaping, and matches on set keys only.
    Scope 去空白 id、构造带转义的 Mem0 范围键，并仅按已设置键匹配。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on models.Scope.
        对 models.Scope 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    scope = models.Scope(user_id=" alice ", run_id=123)
    assert scope.user_id == "alice" and scope.run_id == "123" and scope.agent_id is None
    assert scope.as_filters() == {"user_id": "alice", "run_id": "123"}
    assert models.Scope(user_id="a&b=c%", run_id="r").key() == "run_id=r&user_id=a%26b%3Dc%25"
    assert models.Scope(**SCOPE4).key() == "agent_id=coder&run_id=run-billing&user_id=alice"
    assert models.Scope(user_id="alice").matches(_scope()) and not _scope().matches(models.Scope(user_id="alice"))
    assert models.Scope().is_empty() and not scope.is_empty()
    with pytest.raises(exceptions.ValidationError) as info:
        models.Scope().require_any(context="search")
    assert info.value.error_code == "VALIDATION_001" and info.value.details["context"] == "search"
    assert scope.require_any(context="add") is scope
    for bad in ("", "a b", " "):
        with pytest.raises(pydantic.ValidationError):
            models.Scope(agent_id=bad)


def test_source_event_payload_hashes_golden():
    """SourceEvent normalizes payloads and computes the exact §19.8 sha256 hashes; mismatched hashes are rejected.
    SourceEvent 归一化载荷并计算与 §19.8 完全一致的 sha256；不匹配的哈希被拒绝。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on SourceEvent and event_payload_hash.
        对 SourceEvent 与 event_payload_hash 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    scope = _scope()
    uc = models.SourceEvent(
        object_type="message",
        role="user",
        scope=scope,
        payload={"text": "Do not modify the subscription state machine."},
    )
    assert uc.payload == {"text": "Do not modify the subscription state machine.", "dropped_parts": 0}
    assert uc.payload_hash == "3ed3645c12446c4cc21e746fea721985462031d8c47dd7053029b1abce2014bc"
    assert len(uc.timestamp) == 32 and uc.event_id == "" and uc.visibility == "private"
    e1 = models.SourceEvent(
        object_type="file_diff", role="tool", actor_id="coder", scope=scope,
        payload={
            "path": "src/billing/retry.ts", "patch": "+ backoff = 2 ** attempt",
            "summary": "add exponential backoff", "intent": "fix checkout webhook retry",
        },
    )  # fmt: skip
    e2 = models.SourceEvent(
        object_type="test_log", role="tool", actor_id="coder", scope=scope,
        payload={
            "suite": "billing_e2e", "passed": 11, "failed": 3,
            "failed_tests": [{"name": "subscription_state_regression", "message": "state machine changed"}],
        },
    )  # fmt: skip
    e3 = models.SourceEvent(
        object_type="file_diff", role="tool", actor_id="coder", scope=scope,
        payload={
            "path": "src/billing/idempotency.ts", "patch": "+ idempotencyKey",
            "summary": "add idempotency key", "intent": "fix checkout webhook retry",
        },
    )  # fmt: skip
    e4 = models.SourceEvent(
        object_type="test_log",
        role="tool",
        actor_id="coder",
        scope=scope,
        payload={"suite": "billing_e2e", "passed": 14, "failed": 0},
    )
    assert [e.payload_hash for e in (e1, e2, e3, e4)] == [
        "a27b81718fae86837895813746ec188afca9087544df8950c3b5832742ebfa8d",
        "e51db27bfdf98d59d0abccc1fb59f7413e1f9bbaf1260406fdc2154e4b7ce1e2",
        "cae8b909c617c029c16e12f5ef04e0906be8a797b07eca94132db1654344fb2a",
        "ea5edcbd2c3f0b79ef01e3768590081cf1a024f8081b4c2b2d17155dd68ff272",
    ]
    assert e2.payload["status"] == "failed" and e4.payload["status"] == "passed" and e1.payload["commit"] is None
    assert list(e2.payload) == ["suite", "status", "passed", "failed", "failures", "log", "related_paths"]
    assert (
        models.event_payload_hash(object_type="message", role="user", actor_id=None, payload=uc.payload)
        == uc.payload_hash
    )
    same = models.SourceEvent(**{**uc.model_dump(), "payload_hash": uc.payload_hash})
    assert same.payload_hash == uc.payload_hash
    with pytest.raises(pydantic.ValidationError):
        models.SourceEvent(**{**uc.model_dump(), "payload_hash": "deadbeef"})
    with pytest.raises(pydantic.ValidationError):
        models.SourceEvent(object_type="message", role="extracted", scope=scope, payload={"text": "x"})
    with pytest.raises(pydantic.ValidationError) as info:
        models.SourceEvent(object_type="message", scope=scope, payload={"txt": "x"})
    inner = exceptions.unwrap_hms_error(info.value)
    assert inner.error_code == "VALIDATION_005" and inner.details == {"object_type": "message", "field": "text"}


def test_validate_payload_shapes():
    """validate_payload emits the §4.3 key sets in order, honours aliases and defaults, and rejects bad fields.
    validate_payload 按序输出 §4.3 键集、尊重别名与默认值，并拒绝非法字段。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on models.validate_payload.
        对 models.validate_payload 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    vp = models.validate_payload
    assert list(vp("message", {"text": "hi", "extra": 1})) == ["text", "dropped_parts"]
    tool = vp("tool_call", {"tool": "bash", "arguments": {"cmd": "ls"}, "result": {"ok": True}, "status": "ok"})
    assert list(tool) == ["tool", "args", "result", "status", "target", "intent", "constraints"]
    assert tool["args"] == {"cmd": "ls"} and tool["result"] == '{"ok": true}' and tool["constraints"] == []
    assert list(vp("file_diff", {"path": "a.py", "patch": "+x"})) == [
        "path",
        "patch",
        "summary",
        "intent",
        "commit",
        "constraints",
    ]
    log = vp("test_log", {"suite": "s", "failures": ["t1"]})
    assert (
        log["status"] == "failed"
        and log["failures"] == [{"name": "t1", "message": None}]
        and log["related_paths"] == []
    )
    assert (
        vp("test_log", {"suite": "s"})["status"] == "passed"
        and vp("test_log", {"suite": "s", "status": "failed"})["status"] == "failed"
    )
    sensor = vp("sensor_trace", {"sensor": "temp", "samples": [{"t": "2026-05-01T00:00:00Z", "v": 1.5}]})
    assert list(sensor) == ["sensor", "series_ref", "summary", "samples"] and sensor["samples"] == [
        {"t": "2026-05-01T00:00:00.000000+00:00", "v": 1.5}
    ]
    assert list(vp("feedback", {"target_id": "atm_1", "feedback": "state_wrong"})) == [
        "target_id",
        "feedback",
        "reason",
        "correction",
    ]
    assert vp("feedback", {"target_id": None, "feedback": "state_missing", "correction": {"text": "x"}})[
        "correction"
    ] == {"text": "x"}
    assert vp("checkpoint", {}) == {"label": None}
    assert list(vp("rollback", {"checkpoint_id": "ckp_1"})) == ["checkpoint_id", "label", "reason"]
    system = vp("system", {"op": "revoke", "anchor_id": "anc_1"})
    assert list(system) == [
        "op",
        "memory_id",
        "atom_id",
        "text",
        "metadata",
        "expiration_date",
        "package_id",
        "anchor_id",
    ]
    assert vp("scheduler", {"phases": ["orient"]}) == {"phases": ["orient"]}
    # 'purge' (§16.6) is a system op like delete/revoke; its parameters travel in `metadata`.
    # 'purge'（§16.6）与 delete/revoke 一样是系统 op；其参数放在 `metadata` 里。
    purge = vp("system", {"op": "purge", "metadata": {"older_than_days": 90, "cutoff": "t"}})
    assert purge["op"] == "purge" and purge["metadata"] == {"older_than_days": 90, "cutoff": "t"}
    bad_cases = [
        ("message", {}, "text", None),
        ("message", {"text": "x", "dropped_parts": "1"}, "dropped_parts", None),
        ("tool_call", {"tool": "t", "status": "meh"}, "status", None),
        ("file_diff", {"path": "p"}, "patch", None),
        ("file_diff", {"path": "p", "patch": "x", "constraints": ["ok", 3]}, "constraints", 1),
        ("test_log", {"suite": "s", "failures": [{"message": "no name"}]}, "failures", 0),
        ("test_log", {"suite": "s", "passed": True}, "passed", None),
        ("sensor_trace", {"sensor": "s", "samples": [{"t": "garbage"}]}, "samples", 0),
        ("system", {"op": "explode"}, "op", None),
        ("scheduler", {"phases": "orient"}, "phases", None),
        ("bogus", {}, "object_type", None),
        ("message", "not a dict", "payload", None),
    ]
    for object_type, payload, field, index in bad_cases:
        with pytest.raises(exceptions.ValidationError) as info:
            vp(object_type, payload)
        assert info.value.error_code == "VALIDATION_005", (object_type, payload)
        assert info.value.details["object_type"] == object_type and info.value.details["field"] == field
        assert info.value.details.get("index") == index, (object_type, payload)


def test_memory_atom_public_row_and_expiration():
    """to_public_dict emits the exact §4.1 key order, promotes keys only when present, and never leaks reserved keys.
    to_public_dict 按 §4.1 精确键序输出、仅在存在时提升键，且从不泄漏保留键。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on MemoryAtom.to_public_dict.
        对 MemoryAtom.to_public_dict 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    atom = _atom(valid_to="2027-01-01T00:00:00Z", user_meta={"k": 1}, actor_id="alice")
    row = atom.to_public_dict(score=0.5, score_details={"final_score": 0.5})
    assert list(row) == [
        "id", "memory", "hash", "metadata", "score", "created_at", "updated_at", "user_id", "run_id", "actor_id",
        "expiration_date", "atom_id", "version", "atom_type", "scale_level", "anchor_ids", "valid_from", "valid_to",
        "confidence", "hms", "score_details",
    ]  # fmt: skip
    assert row["id"] == "mem_000001" and row["memory"] == atom.text and row["metadata"] == {"k": 1}
    assert (
        row["expiration_date"] == "2026-12-31"
        and row["anchor_ids"] == ["anc_000002"]
        and row["created_at"] == "2026-04-20T08:00:00.000000+00:00"
    )
    assert list(row["hms"]) == [
        "time_anchor", "entity_ids", "topic_tags", "sentiment", "subject", "predicate", "obs_type", "source_kind",
        "source_id", "source_ep_ids", "extracted_by", "visibility", "archived_reason",
    ]  # fmt: skip
    assert row["hms"]["obs_type"] == "preference_event" and row["hms"]["visibility"] == "private"
    plain = _atom().to_public_dict()
    assert (
        "score" not in plain and "score_details" not in plain and "expiration_date" not in plain and "role" not in plain
    )
    assert plain["metadata"] is None and "agent_id" not in plain and "actor_id" not in plain
    message = _atom(
        atom_type="message",
        scale_level="L1_micro",
        source_kind="message",
        role="user",
        memory_type="procedural_memory",
        attributed_to="user",
    )
    mrow = message.to_public_dict(anchor_ids=["anc_9"])
    assert (
        mrow["role"] == "user"
        and mrow["memory_type"] == "procedural_memory"
        and mrow["attributed_to"] == "user"
        and mrow["anchor_ids"] == ["anc_9"]
    )
    assert list(mrow)[6:11] == ["user_id", "run_id", "role", "attributed_to", "memory_type"]
    assert set(plain) - {"hms", "score", "score_details"} <= models.RESERVED_ATOM_KEYS | {
        "atom_id",
        "version",
        "anchor_ids",
        "valid_from",
        "valid_to",
        "confidence",
        "atom_type",
        "scale_level",
        "created_at",
        "updated_at",
        "hash",
        "metadata",
    }


def test_graph_edge_and_interface_ep_validators():
    """GraphEdge enforces self-edge/boundary/resolution rules; InterfaceEP slots must cite support nodes.
    GraphEdge 强制自环/边界/解决规则；InterfaceEP 槽位必须引用支持节点。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on GraphEdge and InterfaceEP validators.
        对 GraphEdge 与 InterfaceEP 校验器的断言。

    State effect / 状态影响：
        None.
        无。
    """
    scope = _scope()
    base = dict(src_id="a", dst_id="b", edge_type="use", witness_anchor_ids=["anc_1"], created_by="ingest", scope=scope)
    edge = models.GraphEdge(**base)
    assert edge.scale == "L1_micro" and edge.proposer == "rule" and edge.valid and edge.mentions == 1
    with pytest.raises(pydantic.ValidationError, match="self edge"):
        models.GraphEdge(**{**base, "dst_id": "a"})
    with pytest.raises(pydantic.ValidationError, match="witness_edge_id"):
        models.GraphEdge(**{**base, "is_boundary": True, "created_by": "project"})
    assert (
        models.GraphEdge(
            **{**base, "is_boundary": True, "witness_edge_id": "edg_low", "scale": "L2_meso"}
        ).witness_edge_id
        == "edg_low"
    )
    with pytest.raises(pydantic.ValidationError, match="conflict edges"):
        models.GraphEdge(**{**base, "resolved_at": "2026-06-15T10:04:00Z", "resolved_by_anchor_id": "anc_2"})
    with pytest.raises(pydantic.ValidationError, match="resolved_by_anchor_id"):
        models.GraphEdge(**{**base, "edge_type": "conflict", "resolved_at": "2026-06-15T10:04:00Z"})
    resolved = models.GraphEdge(
        **{**base, "edge_type": "conflict", "resolved_at": "2026-06-15T10:04:00Z", "resolved_by_anchor_id": "anc_2"}
    )
    assert resolved.resolved_at == "2026-06-15T10:04:00.000000+00:00"
    ep = _ep()
    good = dict(
        interface_ep_id="iep_1",
        slots=ep.slots,
        slot_support={"u": ["n0"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"], "b": ["n0"], "kappa": ["n0", "n1"]},
        support_node_ids=["n0", "n1"],
        evidence_anchor_ids=["anc_1"],
        scope=scope,
    )
    iep = models.InterfaceEP(**good)
    assert iep.scale == "L2_meso" and iep.cohesion_score == 0.0 and iep.coarsen_op_version == "rule-coarsen-v1"
    with pytest.raises(pydantic.ValidationError, match="cites no support node"):
        models.InterfaceEP(**{**good, "slot_support": {"u": ["n0"]}})
    with pytest.raises(pydantic.ValidationError, match="outside support_node_ids"):
        models.InterfaceEP(**{**good, "slot_support": {**good["slot_support"], "y": ["n9"]}})
    with pytest.raises(pydantic.ValidationError):
        models.InterfaceEP(**{**good, "support_node_ids": ["n0"]})
    with pytest.raises(pydantic.ValidationError):
        models.InterfaceEP(**{**good, "evidence_anchor_ids": []})
    node = models.GraphNode(
        node_id="atm_1", node_kind="ep", ref_id="ep_1", scale="L1_micro", scope=scope, anchor_ids=["anc_1"]
    )
    assert node.valid and node.mentions == 1
    assert (
        models.BoundaryEdge(
            edge_id="e",
            inner_node_id="i",
            outer_node_id="o",
            direction="out",
            edge_type="use",
            witness_anchor_ids=["anc_1"],
        ).direction
        == "out"
    )


def test_transition_from_ep_and_kappa_shape():
    """Transition.from_ep / from_interface_ep produce the §16.4 rows with the uniform κ_t and first-occurrence w_t.
    Transition.from_ep / from_interface_ep 生成 §16.4 的行，κ_t 形状统一且 w_t 保持首次出现顺序。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on Transition and kappa_t_of.
        对 Transition 与 kappa_t_of 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    ep = _ep()
    tr = models.Transition.from_ep(ep, scope=_scope())
    s_t, a_t, s_t1, r_t, kappa, w_t = tr.to_tuple()
    assert (tr.source_kind, tr.source_id, tr.scale) == ("ep", "ep_000001", "L1_micro")
    assert a_t == "patch src/billing/retry.ts: add exponential backoff" and r_t == "negative"
    assert (
        s_t["u"] == "fix checkout webhook retry" and s_t["ep_id"] == "ep_000001" and s_t["valid_from"] == ep.time_anchor
    )
    assert list(s_t) == ["u", "b", "kappa", "ep_id", "valid_from", "valid_to"] and list(s_t1) == ["y", "r", "ep_id"]
    assert s_t1["y"] == "billing_e2e failed: subscription_state_regression"
    assert list(kappa) == ["constraints", "visibility", "permission_scope", "provenance"]
    assert kappa == {
        "constraints": ["Do not modify the subscription state machine."],
        "visibility": "private",
        "permission_scope": "private",
        "provenance": {"evidence": ["anc_000003", "anc_000002"], "ep_id": "ep_000001"},
    }
    assert w_t == ["anc_000003", "anc_000002"] and tr.time_t == tr.time_t1 == "2026-06-15T10:04:00.000000+00:00"
    iep = models.InterfaceEP(
        interface_ep_id="iep_1",
        slots=ep.slots,
        slot_support={"u": ["n0"], "a": ["n0", "n1"], "y": ["n1"], "r": ["n1"], "b": ["n0"], "kappa": ["n0", "n1"]},
        support_node_ids=["n0", "n1"],
        invariants=[models.Invariant(text="inv", support_node_ids=["n0", "n1"], anchor_ids=["anc_1"], confidence=0.8)],
        residuals=[models.Residual(text="res", kind="missing_outcome")],
        evidence_anchor_ids=["anc_1", "anc_2"],
        scope=_scope(),
        created_at="2026-06-15T10:14:00Z",
        cohesion_score=2 / 3,
    )
    itr = models.Transition.from_interface_ep(
        iep, scope=_scope(), time_t="2026-06-15T10:04:00Z", time_t1="2026-06-15T10:14:00Z", permission_scope="team"
    )
    assert itr.source_kind == "interface_ep" and itr.scale == "L2_meso" and itr.w_t == ["anc_1", "anc_2"]
    assert itr.s_t["support_node_ids"] == ["n0", "n1"] and itr.s_t["valid_from"] == "2026-06-15T10:14:00.000000+00:00"
    assert (
        itr.s_t1["invariants"] == ["inv"]
        and itr.s_t1["residuals"] == ["res"]
        and itr.kappa_t["permission_scope"] == "team"
    )
    assert itr.kappa_t["provenance"] == {
        "evidence": ["anc_1", "anc_2"],
        "interface_ep_id": "iep_1",
        "cohesion_score": 2 / 3,
    }
    assert itr.time_t1 == "2026-06-15T10:14:00.000000+00:00"
    assert models.kappa_t_of(
        constraints=("c",), visibility="team", permission_scope="private", provenance={"x": 1}
    ) == {
        "constraints": ["c"],
        "visibility": "team",
        "permission_scope": "private",
        "provenance": {"x": 1},
    }
    with pytest.raises(pydantic.ValidationError) as info:
        models.AxisQuery(aggregation="topk")
    assert exceptions.unwrap_hms_error(info.value).error_code == "VALIDATION_013"
    assert models.AxisQuery(aggregation="topk", text="q").limit == 20 and models.AxisQuery().scope is None


def test_state_package_closure():
    """StatePackage round-trips through JSON and rejects broken witness/evidence closures.
    StatePackage 可经 JSON 往返，并拒绝破损的见证/证据闭包。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on StatePackage, Anchor.evidence_line and Checkpoint/FeedbackEvent construction.
        对 StatePackage、Anchor.evidence_line 与 Checkpoint/FeedbackEvent 构造的断言。

    State effect / 状态影响：
        None.
        无。
    """
    scope = _scope()
    event = models.SourceEvent(
        event_id="evt_1", object_type="message", role="user", scope=scope, payload={"text": "hello"}
    )
    anchor = models.Anchor(
        anchor_id="anc_1",
        source_event_id="evt_1",
        hash=event.payload_hash,
        source_ref="chat://1",
        timestamp="2026-06-15T09:00:00Z",
        ingested_at="2026-09-02T10:00:00Z",
        scope=scope,
    )
    assert anchor.evidence_line() == f"chat://1 sha256:{event.payload_hash[:12]} @2026-06-15T09:00:00.000000+00:00"
    item = models.ContextItem(item_id="atm_1", slot="facts", text="hello", anchor_ids=["anc_1"])
    context = models.ContextIR(query="q", as_of="2026-09-02T10:00:00Z", scope=scope, facts=[item])
    pcir = models.PCIR(
        payload={},
        witness={"items": {"atm_1": ["anc_1"]}},
        human_gloss="",
        proof_obligations=[
            models.ProofObligation(name="evidence_coverage", status="satisfied"),
        ],
    )
    assert pcir.ok()
    assert not models.PCIR(
        payload={},
        witness={},
        human_gloss="",
        proof_obligations=[models.ProofObligation(name="governance", status="violated")],
    ).ok()
    atom = _atom(anchor_ids=["anc_1"], scope=scope)
    package = models.StatePackage(
        package_id="pkg_1", created_at="2026-09-02T10:00:00Z", goal=None, scope=scope, context=context, pcir=pcir,
        atoms=[atom], eps=[], interface_eps=[], edges=[], anchors=[anchor], source_events=[event],
    )  # fmt: skip
    assert package.version == "hms-elite/state-package/1"
    assert models.StatePackage.from_json(package.to_json()) == package
    with pytest.raises(pydantic.ValidationError, match="witness closure"):
        models.StatePackage(**{**package.model_dump(), "anchors": []})
    with pytest.raises(pydantic.ValidationError, match="evidence closure"):
        models.StatePackage(**{**package.model_dump(), "source_events": []})
    edge = models.GraphEdge(
        src_id="a", dst_id="b", edge_type="use", witness_anchor_ids=["anc_missing"], created_by="ingest", scope=scope
    )
    with pytest.raises(pydantic.ValidationError, match="anc_missing"):
        models.StatePackage(**{**package.model_dump(), "edges": [edge.model_dump()]})
    checkpoint = models.Checkpoint(
        anchor_id="anc_1",
        scope=scope,
        atom_state={"atm_1": {"valid_to": None, "archived_reason": None}},
        edge_state={},
        iep_state={},
        history_seq=3,
        snapshot_hash="h",
        created_at="2026-09-02T10:00:00Z",
    )
    assert checkpoint.ep_state == {} and checkpoint.checkpoint_id == ""
    feedback = models.FeedbackEvent(
        target_id=None,
        target_kind="scope",
        feedback="state_missing",
        anchor_id="anc_1",
        scope=scope,
        created_at="2026-09-02T10:00:00Z",
        correction={"text": "x"},
    )
    assert feedback.actions == [] and feedback.feedback == "state_missing"
    with pytest.raises(pydantic.ValidationError):
        models.FeedbackEvent(
            target_kind="atom", feedback="not_a_type", anchor_id="anc_1", scope=scope, created_at="2026-09-02T10:00:00Z"
        )


def test_memory_utils_messages_scope_and_params(caplog):
    """coerce_messages / build_scope_and_metadata / parameter validators / mask_ids behave per §9.5.
    coerce_messages / build_scope_and_metadata / 参数校验器 / mask_ids 的行为遵循 §9.5。

    Input / 输入：
        caplog: pytest log capture fixture.
        caplog：pytest 日志捕获夹具。

    Output / 输出：
        Assertions on hms_elite.memory.utils and MemoryBase.
        对 hms_elite.memory.utils 与 MemoryBase 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert mutils.coerce_messages("hi") == [{"role": "user", "content": "hi"}]
    assert mutils.coerce_messages({"role": "assistant", "content": "x"}) == [{"role": "assistant", "content": "x"}]
    mixed = mutils.coerce_messages(
        [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "url": "x"}, "b"]},
            {"role": "tool", "content": None},
            {"role": "system", "content": "s", "name": "n"},
        ]
    )
    assert mixed == [
        {"role": "user", "content": "a\nb", "dropped_parts": 1},
        {"role": "system", "content": "s", "name": "n"},
    ]
    for bad in (42, [{"role": "user"}], ["text"], [{"content": "x"}]):
        with pytest.raises(exceptions.ValidationError) as info:
            mutils.coerce_messages(bad)
        assert info.value.error_code == "VALIDATION_003"
    assert (
        mutils.parse_messages(
            [
                {"role": "user", "content": "a"},
                {"role": "tool", "content": None},
                {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
            ]
        )
        == "user: a\nassistant: b\n"
    )
    with caplog.at_level(logging.WARNING, logger="hms_elite.memory.utils"):
        scope, meta = mutils.build_scope_and_metadata(
            user_id=" alice ",
            run_id="run-a",
            input_metadata={"k": {"n": 1}, "user_id": "x", "hash": "y", "memory": "z"},
        )
    assert scope == models.Scope(user_id="alice", run_id="run-a") and meta == {"k": {"n": 1}}
    warnings_seen = [rec.getMessage() for rec in caplog.records if "reserved metadata keys" in rec.getMessage()]
    assert len(warnings_seen) == 1 and "hash, memory, user_id" in warnings_seen[0]
    with pytest.raises(exceptions.ValidationError) as info:
        mutils.build_scope_and_metadata()
    assert info.value.error_code == "VALIDATION_001" and info.value.details["provided_ids"] == {
        "user_id": None,
        "agent_id": None,
        "run_id": None,
    }
    with pytest.raises(ValueError):
        mutils.build_scope_and_metadata(user_id="a b")
    with pytest.raises(exceptions.ValidationError):
        mutils.build_scope_and_metadata(user_id="a", input_metadata="nope")
    assert (
        mutils.validate_and_trim_entity_id(None, "user_id") is None
        and mutils.validate_and_trim_entity_id(7, "run_id") == "7"
    )
    with pytest.raises(exceptions.ValidationError) as info:
        mutils.reject_top_level_entity_params({"user_id": "a"}, "search")
    assert info.value.error_code == "VALIDATION_008" and isinstance(info.value, ValueError)
    mutils.reject_top_level_entity_params({"filters": {}}, "search")
    mutils.validate_search_params(0.5, 10)
    mutils.validate_search_params(None, None)
    for threshold, top_k in ((1.5, None), (-0.1, None), (True, None), (None, -1), (None, True), (None, 2.5)):
        with pytest.raises(exceptions.ValidationError) as info:
            mutils.validate_search_params(threshold, top_k)
        assert info.value.error_code == "VALIDATION_009"
    assert mutils.validate_and_trim_query("  hi ") == "hi"
    for bad_query in ("", "   ", None):
        with pytest.raises(exceptions.ValidationError) as info:
            mutils.validate_and_trim_query(bad_query)
        assert info.value.error_code == "VALIDATION_010"
    assert mutils.normalize_expiration_date(None) is None
    assert mutils.normalize_expiration_date("2026-12-31") == "2026-12-31"
    assert mutils.normalize_expiration_date(date(2026, 12, 31)) == "2026-12-31"
    assert (
        mutils.normalize_expiration_date(datetime(2026, 12, 31, 23, 30, tzinfo=timezone(timedelta(hours=-5))))
        == "2027-01-01"
    )
    for bad_date in ("31/12/2026", "2026-13-01", 20261231, "2026-12-31T00:00:00"):
        with pytest.raises(exceptions.ValidationError) as info:
            mutils.normalize_expiration_date(bad_date)
        assert info.value.error_code == "VALIDATION_004"
    assert (
        mutils.estimate_tokens("abcd") == 9 and mutils.estimate_tokens("") == 8 and mutils.estimate_tokens("中文") == 10
    )
    # CJK bound (CONTRACT §9.6): 3 UTF-8 bytes per character → 0.75 estimated tokens per character, never below the
    # ~1 token per 1.5 characters a Chinese tokenizer produces, so Chinese budgets are conservative, not under-counted.
    # CJK 上界（CONTRACT §9.6）：每字 3 个 UTF-8 字节 → 每字 0.75 个估算 token，绝不低于中文分词器约每 1.5 字 1 个 token
    # 的实际值，因此中文预算偏保守而不会少算。
    assert mutils.estimate_tokens("中文" * 10) - 8 >= math.ceil(20 / 1.5) and mutils.estimate_tokens("中文" * 10) == 23
    assert (
        mutils.mask_ids("atm_000001 and mem_abcdef123456 cst_000001 x_000001 atm_12")
        == "<atm> and <mem> <cst> x_000001 atm_12"
    )
    assert repr(mutils._UNSET) == "_UNSET" and not mutils._UNSET
    assert sorted(mbase.MemoryBase.__abstractmethods__) == ["delete", "get", "get_all", "history", "update"]
    assert (
        str(inspect.signature(mbase.MemoryBase.update))
        == "(self, memory_id, text=None, metadata=None, expiration_date=_UNSET, data=None)"
    )
    assert (
        str(inspect.signature(mbase.MemoryBase.get_all))
        == "(self, *, filters=None, top_k=20, show_expired=False, **kwargs)"
    )


def test_bilingual_docstrings():
    """Every hms_elite/tests/examples module and public function follows the §0.3 bilingual layout.
    每个 hms_elite/tests/examples 模块与公开函数都遵循 §0.3 的双语布局。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over module docstrings, header comment pairs and function docstrings.
        对模块文档字符串、头部注释对与函数文档字符串的断言。

    State effect / 状态影响：
        None (imports every module).
        无（导入每个模块）。
    """
    files = (
        sorted((ROOT / "hms_elite").rglob("*.py"))
        + sorted((ROOT / "tests").rglob("*.py"))
        + sorted((ROOT / "examples").glob("*.py"))
    )
    assert files, "no modules found"
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        doc = ast.get_docstring(tree, clean=True)
        assert doc, f"{path}: missing module docstring"
        doc_lines = doc.split("\n")
        assert len(doc_lines) >= 2 and not _CJK.search(doc_lines[0]) and _CJK.search(doc_lines[1]), (
            f"{path}: module docstring must be EN line + ZH line"
        )
        first_stmt_end = tree.body[0].end_lineno if tree.body else 0
        after_doc = "\n".join(source.split("\n")[first_stmt_end:])
        positions = [after_doc.find(header) for header in _HEADER_LINES]
        assert all(p >= 0 for p in positions), (
            f"{path}: missing header lines {[h for h, p in zip(_HEADER_LINES, positions) if p < 0]}"
        )
        assert positions == sorted(positions), f"{path}: header lines out of order"
        module = _load_module(path)
        for qualname, func in _public_functions(module):
            _check_function_doc(qualname, func)


def test_tests_modules_never_skip_at_import():
    """No tests module calls pytest.importorskip at module level (it would skip whatever imports every tests file).
    没有 tests 模块在模块层调用 pytest.importorskip（那会连带跳过导入所有 tests 文件的测试）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertion listing every offending `tests/*.py:line`.
        列出每个违规 `tests/*.py:line` 的断言。

    State effect / 状态影响：
        None (parses the tests directory).
        无（解析 tests 目录）。
    """

    def _is_importorskip(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "importorskip"
        )

    offenders: list[str] = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for stmt in tree.body:
            value = getattr(stmt, "value", None)
            targets = [value] if value is not None else []
            if isinstance(stmt, ast.Assign):
                targets = [stmt.value]
            for target in targets:
                if any(_is_importorskip(n) for n in ast.walk(target)):
                    offenders.append(f"{path.relative_to(ROOT)}:{stmt.lineno}")
    assert offenders == [], f"module-level pytest.importorskip in tests: {offenders}"


# ---------------------------------------------------------------------------
# Group A2: schema + StateStore (CONTRACT §18.3 tests 11-15)
# 分组 A2：schema + StateStore（CONTRACT §18.3 测试 11-15）
# ---------------------------------------------------------------------------
def _seed_event(store, text: str, ts: str, scope: models.Scope | None = None, **kw):
    """Insert one message event and its anchor; return (event, anchor).
    插入一条消息事件及其锚点；返回 (event, anchor)。
    """
    ev = models.SourceEvent(scope=scope or _scope(), role="user", payload={"text": text}, timestamp=ts, **kw)
    stored, _ = store.insert_source_event(ev)
    return stored, store.get_or_create_anchor(stored)


def _seed_atom(store, text: str, anchor_ids: list[str], **overrides) -> models.MemoryAtom:
    """Insert a state atom witnessed by anchor_ids.
    插入由 anchor_ids 见证的 state 原子。
    """
    base = dict(
        atom_type="state",
        content={"text": text},
        text=text,
        scale_level="state",
        source_kind="observation",
        source_id="obs_" + text[:4],
        extracted_by="rule@obs-v1",
        scope=_scope(),
        anchor_ids=anchor_ids,
        obs_type="constraint",
        subject="project",
        predicate="must_not",
        time_anchor="2026-06-15T09:00:00Z",
    )
    base.update(overrides)
    return store.insert_atom(models.MemoryAtom(**base))


def test_schema_migrations_and_capabilities(store, tmp_path, clock, ids):
    """Fresh DB at user_version SCHEMA_VERSION (3), 22 base tables + 1 virtual table; re-open idempotent; newer DB → STORE_002; DDL details.
    新库 user_version 为 SCHEMA_VERSION（3）且有 22 张基表 + 1 张虚拟表；重开幂等；更新的库 → STORE_002；DDL 细节。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        tmp_path: pytest temporary directory for a file database.
        tmp_path：pytest 临时目录，用于文件数据库。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        ids: SequentialIds fixture.
        ids：SequentialIds 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Creates a SQLite file under tmp_path.
        在 tmp_path 下创建 SQLite 文件。
    """
    assert schema.SCHEMA_VERSION == 3 and schema.current_version(store._conn) == 3
    assert store.capabilities["schema_version"] == 3
    assert "json1" in store.capabilities and "sqlite_version" in store.capabilities and "wal" in store.capabilities
    names = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    # 22 base tables + 1 FTS5 virtual table (its five shadow tables are SQLite's, listed by name only).
    # 22 张基表 + 1 张 FTS5 虚拟表（它的五张影子表属于 SQLite，只按名字列出）。
    assert len(schema.TABLES) == 22 and schema.VIRTUAL_TABLES == ("atom_fts",) and "compiled_states" in names
    assert "fts5" in store.capabilities and store.capabilities["fts5"] is schema.fts5_available()
    shadows = set(schema.FTS_SHADOW_TABLES) | set(schema.VIRTUAL_TABLES) if store.capabilities["fts5"] else set()
    assert names == set(schema.TABLES) | shadows and len(schema.FTS_SHADOW_TABLES) == 5
    assert set(store.stats()) == set(schema.TABLES)
    history_cols = [r[1] for r in store._conn.execute("PRAGMA table_info(history)")]
    assert history_cols[:10] == [
        "id",
        "memory_id",
        "old_memory",
        "new_memory",
        "event",
        "created_at",
        "updated_at",
        "is_deleted",
        "actor_id",
        "role",
    ]
    assert history_cols[10:] == ["anchor_id", "atom_id", "edge_id", "successor_id", "details_json", "seq"]
    # source_events.role CHECK excludes the synthetic 'extracted' role (enum narrowing #1); eps.scale is the literal.
    # source_events.role 的 CHECK 排除合成角色 'extracted'（枚举收窄 #1）；eps.scale 固定为该字面量。
    ddl = schema.ddl_text()
    assert "role            TEXT CHECK (role IS NULL OR role IN ('user','assistant','system','tool'))" in ddl
    # memory_atoms.role 仍然保留它
    assert "'extracted'" in ddl  # memory_atoms.role keeps it
    assert "scale           TEXT NOT NULL CHECK (scale = 'L1_micro')" in ddl
    assert "CREATE UNIQUE INDEX ux_atoms_head ON memory_atoms(memory_id) WHERE archived_reason IS NULL" in ddl
    # Append-only ladder: 1 = the 0.1 schema, 2 = the retention gate, 3 = atom_fts; build_ddl() is the current
    # (version 3) schema, and the FTS statement exists only from version 3 on.
    # 只追加的阶梯：1 = 0.1 版模式，2 = 保留策略清理门，3 = atom_fts；build_ddl() 是当前（版本 3）模式，
    # FTS 语句只从版本 3 起出现。
    assert list(schema.MIGRATIONS) == [1, 2, 3] and schema.MIGRATIONS[1] == schema.build_ddl(1)
    assert schema.MIGRATIONS[2] == schema.upgrade_2() and schema.build_ddl() == schema.build_ddl(3)
    assert schema.MIGRATIONS[3] == schema.upgrade_3() == [schema.FTS_DDL] and list(schema.POST_MIGRATIONS) == [3]
    assert schema.build_ddl(2) == schema.build_ddl(3)[:-1] and schema.FTS_DDL not in schema.build_ddl(2)
    assert "'PURGE'" in ddl and "'PURGE'" not in "\n".join(schema.build_ddl(1))
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO source_events(event_id, source_system, object_type, event_type, role, timestamp, ingested_at, "
            "visibility, user_id, scope_key, payload_json, payload_hash) "
            "VALUES ('e','hms','message','created','extracted','t','t','private','alice','user_id=alice','{}','h')"
        )
    # ux_atoms_head: a second non-archived version in one lineage is rejected.
    # ux_atoms_head：同一 lineage 中出现第二个未归档版本会被拒绝。
    _, anc = _seed_event(store, "hello", "2026-06-15T09:00:00Z")
    v1 = _seed_atom(store, "v1", [anc.anchor_id])
    with pytest.raises(exceptions.StoreError) as info:
        _seed_atom(store, "v2", [anc.anchor_id], memory_id=v1.memory_id, version=2)
    assert info.value.error_code == "STORE_001" and "memory_id" in info.value.message
    assert store.resolve_head(v1.memory_id).atom_id == v1.atom_id and len(store.list_versions(v1.memory_id)) == 1
    # File database: WAL, re-open idempotent, newer user_version → STORE_002.
    # 文件数据库：WAL、重新打开幂等、user_version 更高 → STORE_002。
    path = str(tmp_path / "hms.db")
    first = mstorage.StateStore(path, now=clock.now, id_factory=ids)
    assert first.capabilities["wal"] is True and first.capabilities["schema_version"] == schema.SCHEMA_VERSION
    first.set_meta("k", "v")
    first.close()
    second = mstorage.StateStore(path)
    assert second.get_meta("k") == "v" and second.stats()["meta"] >= 2
    second.close()
    # 幂等
    second.close()  # idempotent
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()
    with pytest.raises(exceptions.StoreError) as info:
        mstorage.StateStore(path)
    assert info.value.error_code == "STORE_002" and info.value.suggestion == "pip install -U hms-elite"
    assert schema.detect_capabilities(store._conn)["json1"] == store.capabilities["json1"]
    store.reset()
    assert schema.current_version(store._conn) == schema.SCHEMA_VERSION and store.stats()["memory_atoms"] == 0
    assert store.get_meta("history_seq") is None and store.get_meta("schema_created_at") == clock.now()


def test_fresh_database_is_created_directly_at_schema_version(tmp_path):
    """A version-0 database is built from build_ddl(SCHEMA_VERSION) in one commit, with the ladder's sqlite_master rows.
    版本 0 的库按 build_ddl(SCHEMA_VERSION) 一次提交建成，sqlite_master 行集合与阶梯结果相同。

    Input / 输入：
        tmp_path: pytest temporary directory for the fresh file and the version-1 control file.
        tmp_path：存放新文件与版本 1 对照文件的 pytest 临时目录。

    Output / 输出：
        Assertions on the traced statements (one BEGIN/COMMIT, no history rename), user_version, freelist and the
        sqlite_master row set versus a version-1 file upgraded through the ladder.
        对追踪到的语句（一次 BEGIN/COMMIT、无 history 重命名）、user_version、freelist 以及与经阶梯升级的版本 1 文件
        的 sqlite_master 行集合的断言。

    State effect / 状态影响：
        Creates two SQLite files under tmp_path.
        在 tmp_path 下创建两个 SQLite 文件。
    """
    master = "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
    fresh = sqlite3.connect(str(tmp_path / "fresh.db"), isolation_level=None)
    fresh.execute("PRAGMA foreign_keys = ON")
    executed: list[str] = []
    fresh.set_trace_callback(executed.append)
    assert schema.apply_migrations(fresh, created_at="2026-06-15T09:00:00.000000+00:00") == schema.SCHEMA_VERSION
    fresh.set_trace_callback(None)
    # one transaction: the ladder would BEGIN/COMMIT once per version and rename/copy/drop the history table
    # 只有一个事务：走阶梯的话每个版本各 BEGIN/COMMIT 一次，还会重命名/复制/删除 history 表
    assert executed.count("BEGIN IMMEDIATE") == 1 and executed.count("COMMIT") == 1
    assert not any("history_v1" in stmt for stmt in executed)
    assert sum(stmt in schema.build_ddl(schema.SCHEMA_VERSION) for stmt in executed) == len(schema.build_ddl())
    assert schema.current_version(fresh) == schema.SCHEMA_VERSION
    assert fresh.execute("PRAGMA freelist_count").fetchone()[0] == 0
    assert fresh.execute("SELECT value FROM meta WHERE key = 'schema_created_at'").fetchone()[0].startswith("2026-06-15")
    # a version-1 file upgraded through the ladder has the same schema rows (type, name, tbl_name, sql)
    # 经阶梯升级的版本 1 文件拥有相同的模式行（type, name, tbl_name, sql）
    old = sqlite3.connect(str(tmp_path / "v1.db"), isolation_level=None)
    old.execute("PRAGMA foreign_keys = ON")
    old.execute("BEGIN")
    for stmt in schema.build_ddl(1):
        old.execute(stmt)
    old.execute("PRAGMA user_version = 1")
    old.execute("COMMIT")
    assert schema.apply_migrations(old) == schema.SCHEMA_VERSION
    assert old.execute(master).fetchall() == fresh.execute(master).fetchall()
    # the ladder path frees pages at DROP TABLE history_v1 (step 2) and re-uses them for atom_fts (step 3): it never
    # ends up more compact than the direct build, which has nothing to free
    # 阶梯路径在 DROP TABLE history_v1（第 2 级）时释放页并在 atom_fts（第 3 级）复用它们：它绝不会比直接建库
    # 更紧凑，而直接建库没有可释放的页
    assert old.execute("PRAGMA page_count").fetchone()[0] >= fresh.execute("PRAGMA page_count").fetchone()[0]
    assert old.execute("PRAGMA freelist_count").fetchone()[0] >= 0
    old.close()
    fresh.close()


def test_store_invariants_at_ddl(store, clock):
    """Witness NOT NULL, append-only triggers (STORE_003), mutable columns, witness pre-checks, boundary CHECK, MERGE.
    见证非空、只追加触发器（STORE_003）、可变列、见证预检、边界 CHECK、MERGE。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes the in-memory store only.
        仅写入内存存储。
    """
    _, anc1 = _seed_event(store, "one", "2026-06-15T09:00:00Z")
    _, anc2 = _seed_event(store, "two", "2026-06-15T10:00:00Z")
    _, anc3 = _seed_event(store, "three", "2026-06-15T11:00:00Z")
    a = _seed_atom(store, "atom a", [anc1.anchor_id])
    b = _seed_atom(store, "atom b", [anc2.anchor_id])
    ep = store.insert_ep(
        models.EP(
            slots=models.Slots(u="u", a="a", y="failed", r="negative"),
            time_anchor="2026-06-15T10:04:00Z",
            anchor_ids=[anc2.anchor_id],
            scope=_scope(),
        )
    )
    store.add_history(
        memory_id=a.memory_id,
        old_memory=None,
        new_memory=a.text,
        event="ADD",
        created_at=a.created_at,
        updated_at=a.created_at,
        anchor_id=anc1.anchor_id,
        atom_id=a.atom_id,
    )
    # (a) NOT NULL primary_anchor_id at the DDL level (raw SQL bypassing the store).
    # (a) DDL 层面的 primary_anchor_id NOT NULL（用绕过 store 的原始 SQL）。
    with pytest.raises(sqlite3.IntegrityError) as raw:
        store._conn.execute(
            "INSERT INTO memory_atoms(atom_id, memory_id, atom_type, content_json, text, hash, valid_from, "
            "scale_level, "
            "role, source_kind, source_id, confidence, extracted_by, visibility, user_id, created_at, updated_at) "
            "VALUES ('x','m','state','{}','t','h','t','state','user','observation','o',0.5,'e','private','alice',"
            "'t','t')"
        )
    assert "primary_anchor_id" in str(raw.value)
    # (b) DELETE / content UPDATE on the evidence and state tables → STORE_003 (trigger RAISE).
    # (b) 对证据表与状态表的 DELETE / 内容 UPDATE → STORE_003（触发器 RAISE）。
    for sql in (
        "DELETE FROM anchors",
        "UPDATE anchors SET hash = 'x'",
        "DELETE FROM memory_atoms",
        "UPDATE memory_atoms SET text = 'x'",
        "DELETE FROM eps",
        "UPDATE eps SET u = 'x'",
        "DELETE FROM history",
        "UPDATE history SET event = 'DELETE'",
        "DELETE FROM source_events",
        "UPDATE source_events SET raw_ref = 'x'",
    ):
        with pytest.raises(exceptions.StoreError) as info:
            store._execute(sql)
        assert info.value.error_code == "STORE_003" and info.value.details["trigger"].startswith("HMS_INVARIANT")
    # (c) the mutable columns still work.
    # (c) 可变列仍然可写。
    store.set_valid_to(a.atom_id, "2027-01-01T00:00:00Z", at=clock.now())
    store.set_atom_confidence(a.atom_id, 0.42, at=clock.now())
    got = store.get_atom(a.atom_id)
    assert got.valid_to == "2027-01-01T00:00:00.000000+00:00" and got.confidence == 0.42
    assert got.expiration_date() == "2026-12-31"
    # (d) witness pre-checks before any SQL: dict path, model_construct path (and SupportGraph when group E exists).
    # (d) 任何 SQL 之前的见证预检查：dict 路径、model_construct 路径（以及 group E 存在时的 SupportGraph）。
    store.trace.reset()
    with pytest.raises(exceptions.WitnessMissingError) as info:
        store.insert_edge(
            {
                "src_id": a.atom_id,
                "dst_id": b.atom_id,
                "edge_type": "use",
                "witness_anchor_ids": [],
                "created_by": "ingest",
            }
        )
    assert info.value.details == {"object": "edge", "id": None, "missing": []} and store.trace.tables() == set()
    with pytest.raises(exceptions.WitnessMissingError):
        store.insert_edge(
            models.GraphEdge.model_construct(
                witness_anchor_ids=[],
                src_id=a.atom_id,
                dst_id=b.atom_id,
                edge_type="use",
                created_by="ingest",
                scope=_scope(),
            )
        )
    assert store.trace.tables() == set()
    # group E 路径，等它存在后才会真正执行
    if importlib.util.find_spec("hms_elite.graph") is not None:  # group E path, exercised once it exists
        from hms_elite.configs.base import GraphConfig
        from hms_elite.graph.support_graph import SupportGraph

        graph = SupportGraph(store, GraphConfig())
        with pytest.raises(exceptions.WitnessMissingError):
            graph.insert_edge(
                src_id=a.atom_id, dst_id=b.atom_id, edge_type="use", witness_anchor_ids=[], created_by="ingest"
            )
        assert store.trace.tables() == set()
    with pytest.raises(exceptions.WitnessMissingError) as info:
        store.insert_edge(
            {
                "src_id": a.atom_id,
                "dst_id": b.atom_id,
                "edge_type": "use",
                "witness_anchor_ids": [anc1.anchor_id, "anc_nope"],
                "created_by": "ingest",
            }
        )
    assert info.value.details["missing"] == ["anc_nope"] and info.value.error_code == "WITNESS_001"
    assert store.stats()["graph_edges"] == 0
    with pytest.raises(exceptions.WitnessMissingError):
        store.insert_atom(models.MemoryAtom.model_construct(anchor_ids=[], atom_id="atm_zzz"))
    with pytest.raises(exceptions.WitnessMissingError) as info:
        store.upsert_transition(models.Transition.model_construct(w_t=[], transition_id=""))
    assert info.value.details["object"] == "transition"
    # (e) boundary edge: without witness_edge_id the DDL CHECK (and the model validator) reject it; with it the
    #     AFTER INSERT trigger writes the edge_projections row.
    # (e) 边界边：不带 witness_edge_id 时 DDL CHECK（以及模型校验器）会拒绝；带上它后
    #     AFTER INSERT 触发器会写入 edge_projections 行。
    low = store.insert_edge(
        {
            "src_id": a.atom_id,
            "dst_id": b.atom_id,
            "edge_type": "use",
            "witness_anchor_ids": [anc1.anchor_id],
            "created_by": "ingest",
            "confidence": 0.8,
        }
    )
    with pytest.raises(pydantic.ValidationError):
        models.GraphEdge(
            src_id=b.atom_id,
            dst_id=a.atom_id,
            edge_type="use",
            witness_anchor_ids=[anc1.anchor_id],
            created_by="project",
            is_boundary=True,
            scope=_scope(),
        )
    with pytest.raises(exceptions.StoreError) as info:
        store._execute(
            "INSERT INTO graph_edges(edge_id, src_id, dst_id, edge_type, scale, is_boundary, confidence, "
            "primary_witness_anchor_id, created_by, proposer, user_id, created_at) "
            "VALUES ('edg_raw', ?, ?, 'use', 'L2_meso', 1, 0.5, ?, 'project', 'project-v1', 'alice', 't')",
            (b.atom_id, a.atom_id, anc1.anchor_id),
        )
    assert info.value.error_code == "STORE_001" and "CHECK" in info.value.message
    high = store.insert_edge(
        {
            "src_id": b.atom_id,
            "dst_id": a.atom_id,
            "edge_type": "use",
            "scale": "L2_meso",
            "witness_anchor_ids": [anc1.anchor_id],
            "created_by": "project",
            "is_boundary": True,
            "witness_edge_id": low,
            "proposer": "project-v1",
        }
    )
    assert store.projections_of(high) == [low] and store.projections_from(low) == [high]
    assert store.get_edge(high).is_boundary is True and store.get_edge(high).scale == "L2_meso"
    assert store.set_edge_label(high, "projected", at=clock.now()) is None and store.get_edge(high).label == "projected"
    # (f) MERGE on (src, dst, type, scale): mentions bumped, confidence max, witnesses unioned, label kept.
    # (f) 按 (src, dst, type, scale) 做 MERGE：mentions 递增、confidence 取最大、witnesses 取并集、label 保留。
    again = store.insert_edge(
        models.GraphEdge(
            src_id=a.atom_id,
            dst_id=b.atom_id,
            edge_type="use",
            witness_anchor_ids=[anc3.anchor_id],
            created_by="consolidation",
            confidence=0.5,
            label="ignored",
            scope=_scope(),
        )
    )
    merged = store.get_edge(low)
    assert again == low and merged.mentions == 2 and merged.confidence == 0.8 and merged.label == ""
    assert merged.witness_anchor_ids == [anc1.anchor_id, anc3.anchor_id] == store.edge_witnesses(low)
    assert (
        store.find_edge(a.atom_id, b.atom_id, "use").edge_id == low
        and store.find_edge(a.atom_id, b.atom_id, "causal") is None
    )
    # (g) add_atom_anchor on the primary witness is a no-op; a new support anchor is appended after the primary.
    # (g) 对主见证调用 add_atom_anchor 是空操作；新的支持锚点追加在主锚点之后。
    rows_before = store.stats()["atom_anchor"]
    assert store.add_atom_anchor(a.atom_id, anc1.anchor_id) is False and store.stats()["atom_anchor"] == rows_before
    assert store.add_atom_anchor(a.atom_id, anc2.anchor_id) is True
    assert store.atom_anchor_ids(a.atom_id) == [anc1.anchor_id, anc2.anchor_id]
    store.dispute_atom_anchor(a.atom_id, anc2.anchor_id, at=clock.now())
    assert store.atom_anchor_ids(a.atom_id) == [anc1.anchor_id]
    assert store.atom_anchor_ids(a.atom_id, include_disputed=True) == [anc1.anchor_id, anc2.anchor_id]
    assert store.atoms_for_anchor(anc2.anchor_id) == sorted([a.atom_id, b.atom_id])
    # (h) conflict-only resolution columns and the invariant on EP slots.
    # (h) 仅冲突场景使用的 resolution 列，以及 EP 槽位上的不变式。
    with pytest.raises(exceptions.StoreError):
        store.resolve_conflict(low, at=clock.now(), by_anchor_id=anc1.anchor_id)
    assert store.get_ep(ep.ep_id).slots.y == "failed"


def test_purge_gate_and_schema_migration_2(store, clock, ids, tmp_path):
    """BEFORE DELETE gate opens only inside purge_rows(); a stale flag clears at open; migration 1→2 rebuilds history.
    BEFORE DELETE 门只在 purge_rows() 内打开；残留标志在打开时被清除；迁移 1→2 重建 history。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        ids: SequentialIds fixture.
        ids：SequentialIds 夹具。

        tmp_path: pytest temporary directory for the version-1 file database.
        tmp_path：pytest 临时目录，用于版本 1 的文件库。

    Output / 输出：
        Assertions on STORE_003 with and without the flag, on purge_rows' cleanup and on the upgraded schema.
        对有/无标志时的 STORE_003、purge_rows 的清理与升级后模式的断言。

    State effect / 状态影响：
        Creates SQLite files under tmp_path.
        在 tmp_path 下创建 SQLite 文件。
    """
    # (a) Seven gated triggers, all present with the WHEN clause; every other trigger is untouched.
    # (a) 七个设门触发器都带 WHEN 子句；其余触发器保持原样。
    names = {n for n, _t, _m in schema.PURGE_GATED_TRIGGERS}
    assert names == {
        "trg_events_no_delete",
        "trg_anchors_no_delete",
        "trg_obs_no_delete",
        "trg_eps_no_delete",
        "trg_atoms_no_delete",
        "trg_atom_anchor_no_delete",
        "trg_nodes_no_delete",
    }
    triggers = {r[0]: r[1] for r in store._conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'")}
    assert len(triggers) == 22 and all(schema.PURGE_GATE.strip() in triggers[n] for n in names)
    assert all("WHEN" not in sql for n, sql in triggers.items() if n not in names)
    # (b) Without the flag every DELETE is STORE_003; with it (only purge_rows writes it) the row can go.
    # (b) 没有标志时任何 DELETE 都是 STORE_003；有标志时（只有 purge_rows 会写它）行才能删除。
    _ev, anc = _seed_event(store, "gate", "2026-06-15T09:00:00Z")
    atom = _seed_atom(store, "gated atom", [anc.anchor_id])
    with pytest.raises(exceptions.StoreError) as info:
        store._execute("DELETE FROM atom_anchor WHERE atom_id = ?", (atom.atom_id,))
    assert info.value.error_code == "STORE_003"
    store.set_meta(schema.PURGE_FLAG_KEY, "1")
    store._execute("DELETE FROM atom_anchor WHERE atom_id = ?", (atom.atom_id,))
    assert store.atom_anchor_ids(atom.atom_id) == []
    with store.transaction():
        store._execute("DELETE FROM meta WHERE key = ?", (schema.PURGE_FLAG_KEY,))
    with pytest.raises(exceptions.StoreError) as info:
        store._execute("DELETE FROM memory_atoms WHERE atom_id = ?", (atom.atom_id,))
    assert info.value.error_code == "STORE_003"
    # purge_rows: flag written and removed inside one transaction; the atom row, its links and its source row are gone.
    # purge_rows：标志在一个事务内写入并移除；原子行、其链接与来源行都已删除。
    store.archive_atom(atom.atom_id, reason="deleted", at=clock.now())
    removed = store.purge_rows([atom.atom_id])
    assert removed["atoms"] == [atom.atom_id] and removed["anchors"] == [] and removed["events"] == []
    assert store.get_atom(atom.atom_id) is None and store.get_meta(schema.PURGE_FLAG_KEY) is None
    assert store.get_anchor(anc.anchor_id) is not None and store.purge_rows([]) == {
        "atoms": [], "observations": [], "eps": [], "anchors": [], "events": [], "nodes": 0, "vectors": 0, "links": 0,
    }  # fmt: skip
    # (c) A stale flag left in a file by hand is cleared when the store opens, so the gate is armed again.
    # (c) 手工留在文件里的残留标志在 store 打开时被清除，门重新武装。
    stale = str(tmp_path / "stale.db")
    mstorage.StateStore(stale, now=clock.now, id_factory=ids).close()
    raw = sqlite3.connect(stale)
    raw.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, '1')", (schema.PURGE_FLAG_KEY,))
    raw.commit()
    raw.close()
    reopened = mstorage.StateStore(stale, now=clock.now, id_factory=ids)
    assert reopened.get_meta(schema.PURGE_FLAG_KEY) is None
    _ev2, anc2 = _seed_event(reopened, "stale", "2026-06-15T09:00:00Z")
    with pytest.raises(exceptions.StoreError) as info:
        reopened._execute("DELETE FROM anchors WHERE anchor_id = ?", (anc2.anchor_id,))
    assert info.value.error_code == "STORE_003"
    reopened.close()
    # (d) Migration 1 → 2: a version-1 file (ungated triggers, history CHECK without PURGE) upgrades in place with its
    #     rows intact; the result equals a fresh schema; without the migration a PURGE history row is rejected.
    # (d) 迁移 1 → 2：版本 1 的文件（无门触发器、history CHECK 无 PURGE）原地升级且行保持完整；
    #     结果与全新模式相同；不迁移时 PURGE 历史行会被拒绝。
    old = str(tmp_path / "v1.db")
    raw = sqlite3.connect(old, isolation_level=None)
    raw.execute("PRAGMA foreign_keys = ON")
    raw.execute("BEGIN")
    for stmt in schema.build_ddl(1):
        raw.execute(stmt)
    raw.execute("PRAGMA user_version = 1")
    raw.execute("COMMIT")
    v1 = mstorage.StateStore(":memory:", now=clock.now, id_factory=ids)
    assert "'PURGE'" not in "\n".join(schema.build_ddl(1))
    t = "2026-06-15T09:00:00.000000+00:00"
    raw.execute(
        "INSERT INTO source_events(event_id, source_system, object_type, event_type, timestamp, ingested_at, "
        "visibility, "
        f"user_id, scope_key, payload_json, payload_hash) VALUES ('evt_1','hms','message','created','{t}','{t}',"
        "'private','alice','user_id=alice','{}','h')"
    )
    raw.execute(
        "INSERT INTO anchors(anchor_id, source_event_id, hash, timestamp, ingested_at, permission_scope, user_id) "
        f"VALUES ('anc_1','evt_1','h','{t}','{t}','private','alice')"
    )
    raw.execute(
        "INSERT INTO history(id, memory_id, new_memory, event, created_at, updated_at, anchor_id, seq) "
        "VALUES ('hst_1','mem_1','hello','ADD','t','t','anc_1',1)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            "INSERT INTO history(id, memory_id, event, created_at, updated_at, anchor_id, seq) "
            "VALUES ('hst_2','mem_1','PURGE','t','t','anc_1',2)"
        )
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("DELETE FROM anchors")
    raw.close()
    upgraded = mstorage.StateStore(old, now=clock.now, id_factory=ids)
    assert upgraded.capabilities["schema_version"] == 3 and schema.current_version(upgraded._conn) == 3
    master = "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
    assert upgraded._conn.execute(master).fetchall() == v1._conn.execute(master).fetchall()
    assert upgraded.get_history("mem_1") == [
        {
            "id": "hst_1",
            "memory_id": "mem_1",
            "old_memory": None,
            "new_memory": "hello",
            "event": "ADD",
            "created_at": "t",
            "updated_at": "t",
            "is_deleted": False,
            "actor_id": None,
            "role": None,
            "anchor_id": "anc_1",
        }
    ]
    upgraded.set_meta("history_seq", "1")
    upgraded.add_history(
        memory_id="mem_1", old_memory=None, new_memory=None, event="PURGE", created_at=clock.now(),
        updated_at=clock.now(), is_deleted=1, anchor_id="anc_1",
    )  # fmt: skip
    assert [r["event"] for r in upgraded.get_history("mem_1")] == ["ADD", "PURGE"]
    assert upgraded.stats()["anchors"] == 1 and "history_v1" not in {
        r[0] for r in upgraded._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    upgraded.close()
    v1.close()


def test_store_transaction_rollback_and_trace(store, clock):
    """An exception inside transaction() leaves no rows; trace lists touched tables; nested transactions re-enter.
    transaction() 内的异常不留下任何行；trace 列出触及的表；嵌套事务可重入。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes the in-memory store only.
        仅写入内存存储。
    """
    _, anc = _seed_event(store, "seed", "2026-06-15T09:00:00Z")
    before = store.stats()
    assert store.in_transaction is False
    with pytest.raises(RuntimeError):
        with store.transaction():
            assert store.in_transaction is True
            _seed_atom(store, "rolled back", [anc.anchor_id])
            # 可重入：仍然是同一个外层事务
            with store.transaction():  # re-entrant: still the same outer transaction
                assert store.in_transaction is True
                store.set_meta("inner", "1")
            assert store.in_transaction is True
            raise RuntimeError("boom")
    assert store.in_transaction is False and store.stats() == before and store.get_meta("inner") is None
    # Nested success commits once at the outermost level.
    # 嵌套成功时只在最外层提交一次。
    with store.transaction():
        with store.transaction():
            atom = _seed_atom(store, "kept", [anc.anchor_id])
        assert store.get_atom(atom.atom_id) is not None
    assert store.in_transaction is False and store.stats()["memory_atoms"] == before["memory_atoms"] + 1
    # A failing statement inside an implicit transaction rolls back and maps to StoreError.
    # 隐式事务中失败的语句会回滚并映射为 StoreError。
    with pytest.raises(exceptions.StoreError):
        # ux_atoms_head：第二个存活版本
        store.insert_atom(atom.model_copy(update={"atom_id": ""}))  # ux_atoms_head: second live version
    assert store.stats()["memory_atoms"] == before["memory_atoms"] + 1 and store.in_transaction is False
    # Trace.
    # Trace（追踪）。
    store.trace.reset()
    assert store.trace.tables() == set()
    store.get_atom(atom.atom_id)
    assert store.trace.tables() == {"memory_atoms", "atom_entity", "entities", "atom_topic", "atom_anchor"}
    store.trace.reset()
    store.get_anchor(anc.anchor_id)
    assert store.trace.tables() == {"anchors"}
    store.trace.enabled = False
    store.get_history(atom.memory_id)
    assert store.trace.tables() == {"anchors"}
    store.trace.enabled = True
    _, anc2 = _seed_event(store, "second witness", "2026-06-15T10:00:00Z")
    other = _seed_atom(store, "other", [anc.anchor_id])
    store.trace.reset()
    store.insert_edge(
        {
            "src_id": atom.atom_id,
            "dst_id": other.atom_id,
            "edge_type": "temporal",
            # 第二个见证由 store 自己写入
            "witness_anchor_ids": [anc.anchor_id, anc2.anchor_id],  # the second witness is written by the store itself
            "created_by": "ingest",
        }
    )
    assert {"graph_edges", "edge_witnesses", "anchors", "graph_nodes"} <= store.trace.tables()
    assert "history" not in store.trace.tables()
    # next_seq is transactional and monotonic.
    # next_seq 具备事务性且单调递增。
    assert store.next_seq("counter") == 1 and store.next_seq("counter") == 2
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.next_seq("counter")
            raise RuntimeError("undo")
    assert store.next_seq("counter") == 3
    # Injected clock and ids are the only sources of instants and ids.
    # 注入的时钟与 id 工厂是时间点和 id 的唯一来源。
    assert store.now() == clock.now() and store.new_id("atm").startswith("atm_")
    clock.tick(seconds=60)
    assert store.now() == clock.now() and store.now() != before and len(store.now()) == 32


def test_store_lineage_helpers(store, clock):
    """insert_atom assigns ids/versions; resolve_head, list_versions, archive_atom, restore_atom; version uniqueness.
    insert_atom 分配 id/版本；resolve_head、list_versions、archive_atom、restore_atom；版本唯一性。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes the in-memory store only.
        仅写入内存存储。
    """
    ev1, anc1 = _seed_event(store, "I like oat-milk lattes.", "2026-04-02T08:00:00Z")
    ev2, anc2 = _seed_event(store, "I switched to black coffee.", "2026-04-20T08:00:00Z")
    assert (ev1.event_id, anc1.anchor_id) == ("evt_000001", "anc_000001")
    assert (
        store.insert_source_event(
            models.SourceEvent(
                scope=_scope(),
                role="user",
                payload={"text": "I like oat-milk lattes."},
                timestamp="2026-04-02T08:00:00Z",
            )
        )[1]
        is True
    )
    assert store.stats()["source_events"] == 2 and store.anchor_for_event(ev1.event_id).anchor_id == anc1.anchor_id
    v1 = store.insert_atom(
        models.MemoryAtom(
            atom_type="state",
            content={"text": "User likes oat-milk lattes."},
            text="User likes oat-milk lattes.",
            scale_level="state",
            source_kind="observation",
            source_id="obs_1",
            extracted_by="rule@obs-v1",
            scope=_scope(),
            anchor_ids=[anc1.anchor_id],
            obs_type="preference_event",
            subject="user",
            predicate="likes",
            time_anchor="2026-04-02T08:00:00Z",
            topic_tags=["coffee"],
        ),
        entity_keys=[("latte", "latte", "PROPER", ())],
        topics=["coffee", "preference"],
    )
    assert (v1.atom_id, v1.memory_id, v1.version) == ("atm_000001", "mem_000001", 1)
    assert v1.created_at == clock.now() == v1.updated_at and v1.valid_from == "2026-04-02T08:00:00.000000+00:00"
    assert (
        v1.anchor_ids == [anc1.anchor_id] and v1.entity_ids == ["latte"] and v1.topic_tags == ["coffee", "preference"]
    )
    assert v1.hash == models.MemoryAtom.model_validate(v1.model_dump()).hash and len(v1.hash) == 32
    node = store.get_node(v1.atom_id)
    assert node.node_kind == "observation" and node.ref_id == "obs_1" and node.scale == "state" and node.valid is True
    assert store.resolve_head("mem_000001").atom_id == v1.atom_id and store.latest_version("mem_000001").version == 1
    assert store.find_atoms_by_hash(_scope(), v1.hash)[0].atom_id == v1.atom_id
    assert store.count_atoms(_scope()) == 1 and store.valid_ids(_scope())["atoms"] == [v1.atom_id]
    # Revision: archive v1 at the new event time, insert v2 in the same lineage.
    # 修订：在新的事件时间归档 v1，并在同一 lineage 中插入 v2。
    at = "2026-04-20T08:00:00Z"
    store.archive_atom(v1.atom_id, reason="revised", at=at)
    archived = store.get_atom(v1.atom_id)
    assert archived.archived_reason == "revised" and archived.valid_to == "2026-04-20T08:00:00.000000+00:00"
    assert archived.updated_at == "2026-04-20T08:00:00.000000+00:00" and store.get_node(v1.atom_id).valid is False
    assert store.resolve_head("mem_000001") is None and store.latest_version("mem_000001").atom_id == v1.atom_id
    v2 = store.insert_atom(
        v1.model_copy(
            update={
                "atom_id": "",
                "version": 2,
                "text": "User likes black coffee.",
                "content": {"text": "User likes black coffee."},
                "hash": "",
                "anchor_ids": [anc2.anchor_id],
                "time_anchor": at,
                "valid_from": None,
                "valid_to": None,
                "created_at": None,
                "updated_at": None,
                "archived_reason": None,
            }
        )
    )
    assert (v2.atom_id, v2.memory_id, v2.version) == ("atm_000002", "mem_000001", 2)
    assert store.resolve_head("mem_000001").atom_id == v2.atom_id and store.latest_version("mem_000001").version == 2
    assert [a.version for a in store.list_versions("mem_000001")] == [1, 2]
    assert [a.atom_id for a in store.axis_select(_scope())] == [v2.atom_id]
    assert [a.atom_id for a in store.axis_select(_scope(), include_expired=True)] == [v2.atom_id, v1.atom_id]
    assert [a.atom_id for a in store.axis_select(_scope(), as_of="2026-04-10T00:00:00Z")] == []
    assert store.scope_atom_state(_scope()) == {
        v1.atom_id: {"valid_to": "2026-04-20T08:00:00.000000+00:00", "archived_reason": "revised"},
        v2.atom_id: {"valid_to": None, "archived_reason": None},
    }
    # ux_atoms_lineage_version rejects a duplicate version even when archived.
    # 即使已归档，ux_atoms_lineage_version 仍会拒绝重复的版本号。
    with pytest.raises(exceptions.StoreError) as info:
        store.insert_atom(v2.model_copy(update={"atom_id": "", "archived_reason": "deleted", "valid_to": clock.now()}))
    assert "version" in info.value.message and store.stats()["memory_atoms"] == 2
    # Restore v1 requires archiving v2 first (one head per lineage); cascade back and forth.
    # 恢复 v1 需要先归档 v2（每个 lineage 只有一个 head）；来回级联。
    clock.tick(seconds=60)
    with pytest.raises(exceptions.StoreError):
        store.restore_atom(v1.atom_id, at=clock.now())
    store.archive_atom(v2.atom_id, reason="rollback:ckp", at=clock.now())
    store.restore_atom(v1.atom_id, at=clock.now())
    restored = store.get_atom(v1.atom_id)
    assert restored.archived_reason is None and restored.valid_to is None and restored.updated_at == clock.now()
    assert store.get_node(v1.atom_id).valid is True and store.get_node(v1.atom_id).invalidated_at is None
    assert store.resolve_head("mem_000001").atom_id == v1.atom_id
    assert (
        store.revision_candidates(
            _scope(), subject="user", predicate="likes", obs_type="preference_event", topics=[], objects=[]
        )[0].atom_id
        == v1.atom_id
    )
    assert (
        store.revision_candidates(
            _scope(), subject="user", predicate="hates", obs_type="preference_event", topics=["coffee"], objects=[]
        )[0].atom_id
        == v1.atom_id
    )
    assert (
        store.revision_candidates(
            _scope(), subject="user", predicate="hates", obs_type="preference_event", topics=[], objects=[]
        )
        == []
    )
    # Metadata / visibility / expiration setters and the public row.
    # metadata / visibility / expiration 的 setter 与公开行。
    store.set_user_meta(v1.atom_id, {"k": 1}, at=clock.now())
    store.set_atom_visibility(v1.atom_id, "team", at=clock.now())
    store.set_valid_to(v1.atom_id, "2026-12-01T00:00:00Z", at=clock.now())
    row = store.get_atom(v1.atom_id).to_public_dict()
    assert row["metadata"] == {"k": 1} and row["hms"]["visibility"] == "team" and row["expiration_date"] == "2026-11-30"
    store.set_valid_to(v1.atom_id, None, at=clock.now())
    assert store.get_atom(v1.atom_id).valid_to is None
    # Every reader hydrates entity_ids / topic_tags / anchor_ids.
    # 每个读取入口都会填充 entity_ids / topic_tags / anchor_ids。
    assert store.get_atoms([v2.atom_id, v1.atom_id, "atm_nope"])[1].entity_ids == ["latte"]
    assert store.list_versions("mem_000001")[0].topic_tags == ["coffee", "preference"]
    assert store.axis_select(_scope(), include_expired=True, order="version ASC")[0].anchor_ids == [anc1.anchor_id]


def test_store_entities_keyed_without_run_id(store, clock):
    """get_or_create_entity keys on (normalized, user_id, agent_id): one id across runs; aliases appended.
    get_or_create_entity 以 (normalized, user_id, agent_id) 为键：跨 run 同一 id；别名追加。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions.
        断言。

    State effect / 状态影响：
        Writes the in-memory store only.
        仅写入内存存储。
    """
    run_a = models.Scope(user_id="alice", run_id="run-a")
    run_b = models.Scope(user_id="alice", run_id="run-b")
    e1 = store.get_or_create_entity(run_a, normalized="sarah", canonical="Sarah", entity_type="PROPER")
    e2 = store.get_or_create_entity(
        run_b, normalized="sarah", canonical="Sarah", entity_type="PROPER", aliases=("sarah k.", "sarah")
    )
    assert e1 == e2 == "ent_000001" and store.stats()["entities"] == 1
    assert store.resolve_entity(models.Scope(user_id="alice"), "sarah") == e1
    assert store.resolve_entity(models.Scope(user_id="alice", agent_id="coder"), "sarah") is None
    assert store.resolve_entity(models.Scope(user_id="bob"), "sarah") is None
    assert store.resolve_entities(run_a, ["sarah", "zzz"]) == {"sarah": e1}
    other = store.get_or_create_entity(
        models.Scope(user_id="alice", agent_id="coder"), normalized="sarah", canonical="Sarah", entity_type="PROPER"
    )
    assert other != e1 and store.stats()["entities"] == 2
    store.get_or_create_entity(
        run_a, normalized="sarah", canonical="Sarah", entity_type="PROPER", aliases=("ms sarah",)
    )
    # Link three atoms from three runs; degree spans runs (HMS §3.5 entity axis spans sessions).
    # 关联来自三个 run 的三个原子；度数跨 run 统计（HMS §3.5 实体轴跨会话）。
    linked = []
    for run in ("run-a", "run-b", "run-c"):
        scope = models.Scope(user_id="alice", run_id=run)
        _, anc = _seed_event(store, f"Sarah in {run}", "2026-05-06T09:00:00Z", scope=scope)
        atom = store.insert_atom(
            models.MemoryAtom(
                atom_type="event",
                content={"text": f"Sarah in {run}"},
                text=f"Sarah in {run}",
                scale_level="event",
                source_kind="observation",
                source_id=f"obs_{run}",
                extracted_by="rule@obs-v1",
                scope=scope,
                anchor_ids=[anc.anchor_id],
                obs_type="temporal_fact",
                subject="Sarah",
                predicate="did",
                time_anchor="2026-05-06T00:00:00Z",
            ),
            entity_keys=[("sarah", "Sarah", "PROPER", ())],
        )
        linked.append(atom.atom_id)
        assert atom.entity_ids == ["sarah"]
    assert store.entity_degree(e1) == 3 and store.entity_linked_atoms(e1) == sorted(linked)
    rows = store.entities_in_scope(models.Scope(user_id="alice"))
    assert [r["normalized_text"] for r in rows] == ["sarah"] and rows[0]["aliases"] == ["sarah k.", "ms sarah"]
    assert rows[0]["entity_id"] == e1 and rows[0]["user_id"] == "alice" and rows[0]["agent_id"] is None
    assert store.entities_in_scope(models.Scope(user_id="alice", run_id="run-b")) == rows
    assert store.entities_in_scope(models.Scope(user_id="alice", agent_id="coder")) == []
    assert store.entities_for_atom(linked[0]) == [
        {"entity_id": e1, "normalized_text": "sarah", "canonical_text": "Sarah", "entity_type": "PROPER"}
    ]
    # Archived atoms drop out of the live link count but keep the degree.
    # 已归档原子退出存活链接计数，但度数保持不变。
    store.archive_atom(linked[0], reason="deleted", at=clock.now())
    assert store.entity_linked_atoms(e1) == sorted(linked[1:]) and store.entity_degree(e1) == 3
    assert store.entity_linked_atoms(e1, valid_only=False) == sorted(linked)
    # Topics and vectors round-trip.
    # 主题与向量的往返读写。
    store.add_atom_topic(linked[1], "work")
    store.add_atom_topic(linked[1], "work")
    assert store.get_atom(linked[1]).topic_tags == ["work"]
    store.upsert_vector(linked[1], [0.6, 0.8], embedder="hash@2")
    assert store.load_vectors([linked[1]])[linked[1]] == pytest.approx([0.6, 0.8]) and store.count_vectors() == 1
    assert store.atoms_without_vectors(models.Scope(user_id="alice"), embedder="hash@2") == sorted(
        [linked[0], linked[2]]
    )
    assert linked[1] in store.atoms_without_vectors(None, embedder="hash@64")
    store.delete_vector(linked[1])
    assert store.count_vectors() == 0 and store.load_vectors(None) == {}


# ---------------------------------------------------------------------------
# 15b. _execute never hands a live cursor across the lock: concurrent readers of one instance stay correct
# 15b. _execute 从不把活动游标交到锁外：同一实例的并发读取方保持正确
# ---------------------------------------------------------------------------
def test_store_rows_are_fetched_under_the_lock(store):
    """Three threads hydrating the same atoms through one StateStore never raise or see a short / mis-hydrated result.
    三个线程通过同一个 StateStore 填充相同原子，绝不抛错，也不会拿到缺行 / 错误填充的结果。

    Input / 输入：
        store: In-memory StateStore fixture (check_same_thread=False, one connection, one lock).
        store：内存 StateStore 夹具（check_same_thread=False，单连接、单锁）。

    Output / 输出：
        Assertions over _execute's return value and every concurrent get_atoms result.
        对 _execute 返回值与每次并发 get_atoms 结果的断言。

    State effect / 状态影响：
        Seeds 40 atoms in the in-memory store; runs three short-lived reader threads (about one second).
        向内存 store 写入 40 个原子；运行三个短命读取线程（约一秒）。
    """
    expected: dict[str, list[str]] = {}
    with store.transaction():
        for i in range(40):
            _, anc = _seed_event(store, f"fact {i}", f"2026-06-15T10:{i:02d}:00Z")
            atom = _seed_atom(store, f"fact {i}", [anc.anchor_id], source_id=f"obs_{i}")
            expected[atom.atom_id] = [anc.anchor_id]
    # Rows are materialized inside the lock: no sqlite3.Cursor ever leaves it, and the result reads like one.
    # 结果行在锁内物化：任何 sqlite3.Cursor 都不会离开锁，且返回值用起来和游标一样。
    rows = store._execute("SELECT atom_id FROM memory_atoms ORDER BY atom_id")
    assert not isinstance(rows, sqlite3.Cursor)
    first = rows.fetchone()
    assert first is not None and first["atom_id"] == min(expected)
    assert len(rows.fetchall()) == 39 and rows.fetchone() is None and list(rows) == []
    failures: list[str] = []
    deadline = time.monotonic() + 1.0

    def reader() -> None:
        """Hydrate every atom until the deadline, recording the first exception or incomplete result.
        在截止时间前反复填充全部原子，记录首个异常或不完整的结果。
        """
        while time.monotonic() < deadline and not failures:
            # The race used to surface as IndexError / pydantic ValidationError / TypeError from a wrong statement.
            # 该竞争过去表现为来自错误语句的 IndexError / pydantic ValidationError / TypeError。
            try:
                atoms = store.get_atoms(list(expected))
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                return
            if len(atoms) != len(expected) or any(a.anchor_ids != expected[a.atom_id] for a in atoms):
                failures.append(f"short or mis-hydrated result: {len(atoms)} atoms")
                return

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


# ---------------------------------------------------------------------------
# Group A3: schema v3 (atom_fts), recall channels, attribute listeners (RETRIEVAL_DIAGNOSIS §5, contract layer)
# 分组 A3：模式 v3（atom_fts）、召回通道、属性监听器（RETRIEVAL_DIAGNOSIS §5，契约层）
# ---------------------------------------------------------------------------
_MASTER_ROWS = "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
_T0 = "2026-06-15T09:00:00.000000+00:00"


def _raw_v2_file(path: str) -> sqlite3.Connection:
    """A version-2 file database built from build_ddl(2) with one alice atom (topic + entity) and one bob atom.
    由 build_ddl(2) 建成的版本 2 文件库，含一个 alice 原子（带主题 + 实体）与一个 bob 原子。
    """
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("PRAGMA foreign_keys = ON")
    raw.execute("BEGIN")
    for stmt in schema.build_ddl(2):
        raw.execute(stmt)
    raw.execute("PRAGMA user_version = 2")
    raw.execute(
        "INSERT INTO source_events(event_id, source_system, object_type, event_type, timestamp, ingested_at, "
        f"visibility, user_id, scope_key, payload_json, payload_hash) VALUES ('evt_1','hms','message','created','{_T0}',"
        f"'{_T0}','private','alice','user_id=alice','{{}}','h')"
    )
    raw.execute(
        "INSERT INTO anchors(anchor_id, source_event_id, hash, timestamp, ingested_at, permission_scope, user_id) "
        f"VALUES ('anc_1','evt_1','h','{_T0}','{_T0}','private','alice')"
    )
    for atom_id, user, text, archived in (
        ("atm_1", "alice", "Sarah shipped the retry.ts backoff patch", None),
        ("atm_2", "bob", "部署 回滚 测试 通过了", None),
        ("atm_3", "alice", "an archived version never enters atom_fts", "revised"),
    ):
        raw.execute(
            "INSERT INTO memory_atoms(atom_id, memory_id, atom_type, content_json, text, hash, time_anchor, valid_from, "
            "scale_level, role, source_kind, source_id, confidence, extracted_by, primary_anchor_id, visibility, "
            "user_id, archived_reason, created_at, updated_at) VALUES (?, ?, 'state', '{}', ?, 'h', ?, ?, 'state', "
            "'user', 'observation', ?, 1.0, 'rule@obs-v1', 'anc_1', 'private', ?, ?, ?, ?)",
            (atom_id, f"mem_{atom_id}", text, _T0, _T0, f"obs_{atom_id}", user, archived, _T0, _T0),
        )
    raw.execute("INSERT INTO atom_topic(atom_id, topic) VALUES ('atm_1', 'billing')")
    raw.execute(
        "INSERT INTO entities(entity_id, normalized_text, canonical_text, entity_type, user_id, created_at) "
        f"VALUES ('ent_1', 'sarah', 'Sarah', 'PROPER', 'alice', '{_T0}')"
    )
    raw.execute("INSERT INTO atom_entity(atom_id, entity_id) VALUES ('atm_1', 'ent_1')")
    raw.execute("COMMIT")
    return raw


def test_schema_v3_backfills_fts_from_a_v2_file(tmp_path, clock, ids):
    """Opening a version-2 file runs upgrade_3 + backfill_fts inside one transaction; the result equals a fresh v3 schema.
    打开版本 2 的文件会在一个事务内执行 upgrade_3 + backfill_fts；结果等于全新的 v3 模式。

    Input / 输入：
        tmp_path: pytest temporary directory for the version-2 file.
        tmp_path：存放版本 2 文件的 pytest 临时目录。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        ids: SequentialIds fixture.
        ids：SequentialIds 夹具。

    Output / 输出：
        Assertions on user_version, the atom_fts rows (live atoms only, tokens == fts_tokens), the sqlite_master rows
        versus a fresh database and on the keyword channel finding the backfilled atom.
        对 user_version、atom_fts 行（只含存活原子，tokens == fts_tokens）、与全新库相比的 sqlite_master 行、
        以及关键词通道能找到回填原子的断言。

    State effect / 状态影响：
        Creates SQLite files under tmp_path.
        在 tmp_path 下创建 SQLite 文件。
    """
    if not schema.fts5_available():  # pragma: no cover - stdlib sqlite ships FTS5 everywhere the suite runs
        pytest.skip("sqlite without FTS5")
    old = str(tmp_path / "v2.db")
    raw = _raw_v2_file(old)
    with pytest.raises(sqlite3.OperationalError):
        raw.execute("SELECT count(*) FROM atom_fts")
    raw.close()
    upgraded = mstorage.StateStore(old, now=clock.now, id_factory=ids)
    assert schema.current_version(upgraded._conn) == 3 and upgraded.capabilities["fts5"] is True
    rows = upgraded._conn.execute("SELECT atom_id, tokens FROM atom_fts ORDER BY atom_id").fetchall()
    assert [r[0] for r in rows] == ["atm_1", "atm_2"]  # the archived atm_3 is not backfilled / 已归档的 atm_3 不回填
    assert rows[0][1] == schema.fts_tokens("Sarah shipped the retry.ts backoff patch", ["billing"], ["sarah"])
    assert "retry.ts" in rows[0][1].split() and "billing" in rows[0][1].split() and "部署" in rows[1][1].split()
    fresh = mstorage.StateStore(str(tmp_path / "fresh.db"), now=clock.now, id_factory=ids)
    assert upgraded._conn.execute(_MASTER_ROWS).fetchall() == fresh._conn.execute(_MASTER_ROWS).fetchall()
    assert schema.POST_MIGRATIONS[3] is schema.backfill_fts and upgraded.stats()["memory_atoms"] == 3
    alice, bob = models.Scope(user_id="alice"), models.Scope(user_id="bob")
    assert [a for a, _s in upgraded.keyword_candidates(alice, ["retry.ts"], limit=5)] == ["atm_1"]
    assert [a for a, _s in upgraded.keyword_candidates(bob, ["部署"], limit=5)] == ["atm_2"]
    assert upgraded.keyword_candidates(alice, ["部署"], limit=5) == []
    # The backfill is idempotent on re-open (no second migration) and a v1 file walks 1 → 2 → 3 to the same rows.
    # 重开时不再回填（没有第二次迁移），版本 1 的文件沿 1 → 2 → 3 得到相同的行。
    upgraded.close()
    reopened = mstorage.StateStore(old, now=clock.now, id_factory=ids)
    assert reopened._conn.execute("SELECT count(*) FROM atom_fts").fetchone()[0] == 2
    reopened.close()
    v1 = sqlite3.connect(str(tmp_path / "v1.db"), isolation_level=None)
    v1.execute("BEGIN")
    for stmt in schema.build_ddl(1):
        v1.execute(stmt)
    v1.execute("PRAGMA user_version = 1")
    v1.execute("COMMIT")
    assert schema.apply_migrations(v1) == 3
    assert v1.execute(_MASTER_ROWS).fetchall() == [tuple(r) for r in fresh._conn.execute(_MASTER_ROWS).fetchall()]
    v1.close()
    fresh.close()


def _channel_atom(text: str, anchor_id: str, n: int, **overrides) -> models.MemoryAtom:
    """A state atom for the channel tests (time_anchor spreads over 2026-01 … 2026-06 by n).
    通道测试用的 state 原子（time_anchor 按 n 分布在 2026-01 … 2026-06）。
    """
    base = dict(
        atom_type="state",
        content={"text": text},
        text=text,
        scale_level="state",
        source_kind="observation",
        source_id=f"obs_{n}",
        extracted_by="rule@obs-v1",
        scope=_scope(),
        anchor_ids=[anchor_id],
        obs_type="fact",
        subject="user",
        predicate="states",
        time_anchor=f"2026-{(n % 6) + 1:02d}-15T09:00:00Z",
    )
    base.update(overrides)
    return models.MemoryAtom(**base)


def test_fts_lifecycle_channels_and_query_plans(store, clock):
    """insert / archive / restore / purge maintain atom_fts; the four recall channels are bounded, ordered and indexed.
    insert / archive / restore / purge 维护 atom_fts；四条召回通道有界、有序且走索引。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

    Output / 输出：
        Assertions on keyword (FTS5 bm25 and the LIKE fallback), entity, recency and axis_ids / count_axis results,
        on EXPLAIN QUERY PLAN (no `SCAN memory_atoms`), on fts_match_query escaping and on the FTS row lifecycle.
        对关键词（FTS5 bm25 与 LIKE 退路）、实体、新近度与 axis_ids / count_axis 结果、EXPLAIN QUERY PLAN
        （没有 `SCAN memory_atoms`）、fts_match_query 转义以及 FTS 行生命周期的断言。

    State effect / 状态影响：
        Writes the in-memory store.
        写入内存 store。
    """
    if not store.capabilities["fts5"]:  # pragma: no cover - stdlib sqlite ships FTS5 everywhere the suite runs
        pytest.skip("sqlite without FTS5")
    _, anc = _seed_event(store, "seed", "2026-06-15T09:00:00Z")
    scope = _scope()
    sarah = store.insert_atom(
        _channel_atom("Sarah joined the billing team in March.", anc.anchor_id, 1),
        entity_keys=[("sarah", "Sarah", "PROPER", ())],
        topics=["Team"],
    )
    patch = store.insert_atom(
        _channel_atom("retry.ts backoff patch failed billing_e2e", anc.anchor_id, 2),
        entity_keys=[("retry.ts", "retry.ts", "FILE", ()), ("sarah", "Sarah", "PROPER", ())],
    )
    cjk = store.insert_atom(_channel_atom("部署 回滚 测试 通过了", anc.anchor_id, 3))
    latte = store.insert_atom(_channel_atom("User likes oat-milk lattes", anc.anchor_id, 4, obs_type="preference_event"))
    bob = store.insert_atom(_channel_atom("Sarah works with bob", anc.anchor_id, 5, scope=models.Scope(user_id="bob")))
    tokens = {r[0]: r[1].split() for r in store._conn.execute("SELECT atom_id, tokens FROM atom_fts")}
    assert set(tokens) == {sarah.atom_id, patch.atom_id, cjk.atom_id, latte.atom_id, bob.atom_id}
    # content tokens are stemmed (billing → bill) exactly like utils.scoring.atom_keyword_tokens
    # 内容词元与 utils.scoring.atom_keyword_tokens 一样经过词干化（billing → bill）
    assert {"sarah", "team", "bill"} <= set(tokens[sarah.atom_id]) and "retry.ts" in tokens[patch.atom_id]
    assert {"部署", "回滚", "测试"} <= set(tokens[cjk.atom_id])
    # (a) keyword channel: bm25 order, bound, scope, CJK, identifiers, blank / quoted tokens.
    # (a) 关键词通道：bm25 顺序、上限、范围、CJK、标识符、空白 / 带引号词元。
    hits = store.keyword_candidates(scope, utext.content_tokens("who is Sarah on the billing team"), limit=10)
    assert [a for a, _s in hits] == [sarah.atom_id, patch.atom_id] and hits[0][1] > hits[1][1] > 0.0
    assert [a for a, _s in store.keyword_candidates(scope, ["sarah"], limit=1)] == [sarah.atom_id]
    assert [a for a, _s in store.keyword_candidates(scope, ["retry.ts", "backoff"], limit=5)] == [patch.atom_id]
    assert [a for a, _s in store.keyword_candidates(scope, utext.content_tokens("部署 回滚"), limit=5)] == [cjk.atom_id]
    assert store.keyword_candidates(scope, [], limit=5) == [] and store.keyword_candidates(scope, [" "], limit=5) == []
    assert store.keyword_candidates(scope, ['sa"rah'], limit=5) == [] and store.keyword_candidates(scope, ["x"], limit=0) == []
    assert mstorage.fts_match_query(["retry.ts", "", 'a"b', "retry.ts"]) == '"retry.ts" OR "a""b"'
    # (b) entity channel: keys resolve within the scope's user/agent identity; newest first (patch: 2026-03), bounded.
    # (b) 实体通道：键在范围的 user/agent 身份内解析；最新优先（patch：2026-03）、有界。
    assert store.entity_candidates(scope, ["sarah", "missing"], limit=10) == [patch.atom_id, sarah.atom_id]
    assert store.entity_candidates(scope, ["sarah"], limit=1) == [patch.atom_id]
    assert store.entity_candidates(models.Scope(user_id="bob"), ["sarah"], limit=10) == []
    assert store.entity_candidates(scope, [], limit=10) == []
    # (c) recency channel and the id-only gate: DEFAULT_ATOM_ORDER, whitelists, LIMIT, COUNT.
    # (c) 新近度通道与只取 id 的门控：DEFAULT_ATOM_ORDER、白名单、LIMIT、COUNT。
    assert store.recent_candidates(scope, limit=2) == [latte.atom_id, cjk.atom_id]
    assert store.recent_candidates(scope, limit=10, obs_type_in=["preference_event"]) == [latte.atom_id]
    assert store.recent_candidates(scope, limit=10, scale_in=["event"]) == []
    assert store.axis_ids(scope, where_sql="memory_atoms.time_anchor >= ?", params=("2026-03-01",)) == [
        latte.atom_id,
        cjk.atom_id,
        patch.atom_id,
    ]
    assert store.axis_ids(scope, order="atom_id ASC", limit=2) == sorted(a.atom_id for a in (sarah, patch, cjk, latte))[:2]
    assert store.count_axis(scope) == 4 and store.count_axis(scope, atom_types=["event"]) == 0
    assert [a.atom_id for a in store.axis_select(scope)] == store.axis_ids(scope)
    # (d) every channel is an index search, never a memory_atoms table scan (EXPLAIN QUERY PLAN on the traced SQL).
    # (d) 每条通道都是索引查找，绝不是 memory_atoms 全表扫描（对追踪到的 SQL 做 EXPLAIN QUERY PLAN）。
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    try:
        store.keyword_candidates(scope, ["sarah", "bill"], limit=5)
        store.entity_candidates(scope, ["sarah"], limit=5)
        store.recent_candidates(scope, limit=5, scale_in=["state"])
        store.axis_ids(scope, where_sql="memory_atoms.time_anchor >= ?", params=("2026-03-01",))
    finally:
        store._conn.set_trace_callback(None)
    assert len(seen) >= 4
    for sql in [s for s in seen if s.startswith("SELECT")]:
        plan = " | ".join(r[3] for r in store._conn.execute("EXPLAIN QUERY PLAN " + sql))
        assert "SCAN memory_atoms" not in plan, (sql, plan)
        # The keyword channel ranks inside atom_fts first (virtual-table index) and touches memory_atoms only by
        # primary key for the ranked window; the entity channel resolves its keys through ux_entities_norm_scope;
        # every statement over memory_atoms searches it through an index.
        # 关键词通道先在 atom_fts 内排序（虚拟表索引），只对排好序的窗口按主键触及 memory_atoms；实体通道经
        # ux_entities_norm_scope 解析实体键；凡触及 memory_atoms 的语句都经索引查找它。
        if "memory_atoms" in sql:
            assert "SEARCH memory_atoms USING" in plan, (sql, plan)
        else:
            assert "VIRTUAL TABLE INDEX" in plan or "USING INDEX" in plan, (sql, plan)
    # (e) archive removes the FTS row, restore rebuilds it, purge deletes it; the LIKE fallback finds the same ids.
    # (e) 归档删除 FTS 行，恢复重建它，清理删除它；LIKE 退路找到相同的 id。
    store.archive_atom(patch.atom_id, reason="revised", at=clock.now())
    assert store.keyword_candidates(scope, ["retry.ts"], limit=5) == []
    assert store._conn.execute("SELECT count(*) FROM atom_fts WHERE atom_id = ?", (patch.atom_id,)).fetchone()[0] == 0
    store.restore_atom(patch.atom_id, at=clock.now())
    assert [a for a, _s in store.keyword_candidates(scope, ["retry.ts"], limit=5)] == [patch.atom_id]
    assert store._conn.execute("SELECT tokens FROM atom_fts WHERE atom_id = ?", (patch.atom_id,)).fetchone()[0] == (
        schema.fts_tokens(patch.text, patch.topic_tags, patch.entity_ids)
    )
    store.capabilities["fts5"] = False
    like = store.keyword_candidates(scope, ["retry.ts", "billing", "部署"], limit=5)  # LIKE matches raw text / LIKE 匹配原文
    assert [a for a, _s in like] == [patch.atom_id, cjk.atom_id, sarah.atom_id] and like[0][1] == 2.0
    assert store.keyword_candidates(scope, ["%"], limit=5) == [] and store.keyword_candidates(scope, ["_e2e"], limit=5) == [
        (patch.atom_id, 1.0)
    ]
    store.capabilities["fts5"] = True
    store.archive_atom(latte.atom_id, reason="deleted", at=clock.now(), valid_to=clock.now())
    assert store.purge_rows([latte.atom_id])["atoms"] == [latte.atom_id]
    assert store._conn.execute("SELECT count(*) FROM atom_fts").fetchone()[0] == 4
    # An atom inserted already archived (package import of a superseded version) gets no FTS row.
    # 以已归档状态插入的原子（交接包导入的被取代版本）没有 FTS 行。
    gone = store.insert_atom(_channel_atom("already archived", anc.anchor_id, 6, archived_reason="revised"))
    assert store._conn.execute("SELECT count(*) FROM atom_fts WHERE atom_id = ?", (gone.atom_id,)).fetchone()[0] == 0


def test_attr_listeners_fire_after_commit_only(store, clock, caplog):
    """archive_atom / restore_atom / set_valid_to notify attr_listeners after COMMIT; a rollback discards the queue.
    archive_atom / restore_atom / set_valid_to 在 COMMIT 后通知 attr_listeners；回滚丢弃队列。

    Input / 输入：
        store: In-memory StateStore fixture.
        store：内存 StateStore 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        caplog: pytest log capture fixture.
        caplog：pytest 日志捕获夹具。

    Output / 输出：
        Assertions on the (atom_id, attrs) sequence, on dispatch timing inside nested transactions, on rollback and
        on a raising listener being logged instead of raised.
        对 (atom_id, attrs) 序列、嵌套事务内的派发时机、回滚以及抛错监听器只记日志不抛出的断言。

    State effect / 状态影响：
        Writes the in-memory store.
        写入内存 store。
    """
    _, anc = _seed_event(store, "seed", "2026-06-15T09:00:00Z")
    atom = _seed_atom(store, "listened", [anc.anchor_id])
    events: list[tuple[str, dict]] = []
    store.attr_listeners.append(lambda atom_id, attrs: events.append((atom_id, attrs)))
    at = clock.now()

    def snapshot() -> dict:
        """The listener payload is the FULL atom_attrs snapshot of the committed row (never a partial diff).
        监听器载荷是已提交行的**完整** atom_attrs 快照（绝不是部分差异）。
        """
        return store.atom_attrs([atom.atom_id])[atom.atom_id]

    store.archive_atom(atom.atom_id, reason="deleted", at=at)
    assert events == [(atom.atom_id, snapshot())] and events[0][1]["archived"] is True and events[0][1]["valid_to"] == at
    assert set(events[0][1]) == set(INDEX_ATTR_KEYS)
    store.restore_atom(atom.atom_id, at=at)
    assert events[-1] == (atom.atom_id, snapshot()) and events[-1][1]["archived"] is False and events[-1][1]["valid_to"] == ""
    store.set_valid_to(atom.atom_id, "2026-12-31T00:00:00Z", at=at)
    assert events[-1] == (atom.atom_id, snapshot()) and events[-1][1]["valid_to"] == "2026-12-31T00:00:00.000000+00:00"
    store.set_valid_to(atom.atom_id, None, at=at)
    assert events[-1] == (atom.atom_id, snapshot()) and events[-1][1]["valid_to"] == "" and len(events) == 4
    # Inside a nested transaction nothing is dispatched until the outermost COMMIT, and one atom changed twice yields
    # ONE callback with the committed snapshot (the intermediate state never was committed truth); a rollback drops
    # everything.
    # 嵌套事务内直到最外层 COMMIT 才派发，且同一原子变化两次只产生**一次**回调、携带已提交快照（中间状态从未成为
    # 已提交的真值）；回滚丢弃全部。
    with store.transaction():
        store.archive_atom(atom.atom_id, reason="deleted", at=at)
        with store.transaction():
            store.restore_atom(atom.atom_id, at=at)
        assert len(events) == 4 and len(store._pending_attrs) == 2
    assert len(events) == 5 and store._pending_attrs == []
    assert events[-1] == (atom.atom_id, snapshot()) and events[-1][1]["archived"] is False
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.archive_atom(atom.atom_id, reason="deleted", at=at)
            raise RuntimeError("abort")
    assert len(events) == 5 and store._pending_attrs == [] and store.get_atom(atom.atom_id).archived_reason is None
    # A listener that raises is logged; the committed write and the other listeners are unaffected.
    # 抛错的监听器被记录日志；已提交的写入与其它监听器不受影响。

    def boom(atom_id: str, attrs: dict) -> None:
        raise RuntimeError("index down")

    store.attr_listeners.insert(0, boom)
    with caplog.at_level(logging.WARNING, logger="hms_elite.memory.storage"):
        store.archive_atom(atom.atom_id, reason="deleted", at=at)
    assert len(events) == 6 and "index attribute listener" in caplog.text
    assert store.get_atom(atom.atom_id).archived_reason == "deleted"
    # Without listeners nothing is queued at all.
    # 没有监听器时根本不排队。
    store.attr_listeners.clear()
    with store.transaction():
        store.restore_atom(atom.atom_id, at=at)
        assert store._pending_attrs == []


def test_fts5_unavailable_falls_back_to_like(tmp_path, clock, ids, monkeypatch, caplog):
    """Without the FTS5 module the schema is built without atom_fts, capabilities say so and the keyword channel uses LIKE.
    没有 FTS5 模块时模式不含 atom_fts，capabilities 如实报告，关键词通道走 LIKE。

    Input / 输入：
        tmp_path: pytest temporary directory (file databases bypass the in-memory template).
        tmp_path：pytest 临时目录（文件库绕过内存模板）。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        ids: SequentialIds fixture.
        ids：SequentialIds 夹具。

        monkeypatch: Forces schema.fts5_available() to False.
        monkeypatch：把 schema.fts5_available() 强制为 False。

        caplog: pytest log capture fixture.
        caplog：pytest 日志捕获夹具。

    Output / 输出：
        Assertions on sqlite_master, capabilities, the warning, and on insert / archive / restore / purge / keyword recall.
        对 sqlite_master、capabilities、警告以及 insert / archive / restore / purge / 关键词召回的断言。

    State effect / 状态影响：
        Creates a SQLite file under tmp_path.
        在 tmp_path 下创建 SQLite 文件。
    """
    monkeypatch.setattr(schema, "fts5_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="hms_elite.memory.schema"):
        plain = mstorage.StateStore(str(tmp_path / "nofts.db"), now=clock.now, id_factory=ids)
    assert "no FTS5" in caplog.text and plain.capabilities["fts5"] is False
    names = {r[0] for r in plain._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert names == set(schema.TABLES) and schema.current_version(plain._conn) == 3
    _, anc = _seed_event(plain, "seed", "2026-06-15T09:00:00Z")
    atom = plain.insert_atom(_channel_atom("retry.ts backoff patch", anc.anchor_id, 1))
    scope = _scope()
    assert plain.keyword_candidates(scope, ["retry.ts"], limit=5) == [(atom.atom_id, 1.0)]
    plain.archive_atom(atom.atom_id, reason="deleted", at=clock.now(), valid_to=clock.now())
    assert plain.keyword_candidates(scope, ["retry.ts"], limit=5) == []
    plain.restore_atom(atom.atom_id, at=clock.now())
    assert plain.keyword_candidates(scope, ["retry.ts"], limit=5) == [(atom.atom_id, 1.0)]
    plain.archive_atom(atom.atom_id, reason="deleted", at=clock.now(), valid_to=clock.now())
    assert plain.purge_rows([atom.atom_id])["atoms"] == [atom.atom_id]
    plain.close()
    # The same file opened with FTS5 available reports the table as absent (capabilities['fts5'] stays False).
    # 同一文件在 FTS5 可用时打开，会报告该表不存在（capabilities['fts5'] 仍为 False）。
    monkeypatch.undo()
    if schema.fts5_available():
        again = mstorage.StateStore(str(tmp_path / "nofts.db"), now=clock.now, id_factory=ids)
        assert again.capabilities["fts5"] is False and again.stats()["memory_atoms"] == 0
        again.close()


def test_retrieval_config_bounds():
    """RetrievalConfig resolves the channel bounds from top_k (Mem0's 4·top_k over-fetch, floors 60 / 30) and is on HMSConfig.
    RetrievalConfig 按 top_k 解析通道上限（Mem0 的 4·top_k 过取，下限 60 / 30），并挂在 HMSConfig 上。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on k_for(), explicit overrides, validation, HMSConfig.retrieval and ScoringConfig compatibility.
        对 k_for()、显式覆盖、校验、HMSConfig.retrieval 与 ScoringConfig 兼容性的断言。

    State effect / 状态影响：
        None.
        无。
    """
    default = cfg.RetrievalConfig()
    assert default.k_for(20) == {"semantic": 80, "keyword": 40, "entity": 40, "recency": 40, "pool": 320}
    assert default.k_for(5) == {"semantic": 60, "keyword": 30, "entity": 30, "recency": 30, "pool": 240}
    assert default.k_for(0)["semantic"] == 60 and default.safety_limit == 50000 and default.hard_gate_pool is True
    custom = cfg.RetrievalConfig(k_semantic=10, k_keyword=3, pool_max_factor=2)
    assert custom.k_for(50) == {"semantic": 10, "keyword": 3, "entity": 100, "recency": 100, "pool": 20}
    with pytest.raises(pydantic.ValidationError):
        cfg.RetrievalConfig(k_semantic=0)
    hms_cfg = cfg.HMSConfig.from_dict({"retrieval": {"k_recency": 7, "safety_limit": 10}})
    assert hms_cfg.retrieval.k_for(1)["recency"] == 7 and hms_cfg.retrieval.safety_limit == 10
    assert cfg.HMSConfig().retrieval == default
    # ScoringConfig keeps its eight §5.1 keys; max_candidates is accepted but no longer truncates anything.
    # ScoringConfig 保留 §5.1 的八个键；max_candidates 仍被接受但不再截断任何东西。
    assert list(cfg.ScoringConfig.model_fields) == [
        "w_semantic",
        "w_keyword",
        "w_entity",
        "w_recency",
        "w_support",
        "half_life_days",
        "max_candidates",
        "identifier_weight",
    ]
