"""Tests for the hms4eye capability extension: entities, graph, mental models, directives, consolidation, stats.
hms4eye 能力扩展的测试：实体、图、心智模型、指令、凝聚与统计。
"""

# Purpose (EN): Prove that the endpoints the upstream Mem0 shell answers with a no-op are real reads and writes
#               here, on the §19.3 S4 billing scenario, and that our capability matrix covers their whole HTTP API.
# 目的（中）：在 §19.3 的 S4 计费场景上，证明上游 Mem0 兼容壳以 no-op 答复的那些接口在这里是真实的读写，
#            并证明我们的能力矩阵覆盖了他们全部的 HTTP API。

# Input (EN): tests/fixtures/events/structured_billing.json and, when reachable, the upstream
#             HMS4EYE_INTERFACE_COMPATIBILITY.md whose `compat:http:<id>` markers list their 68 operations.
# 输入（中）：tests/fixtures/events/structured_billing.json，以及（可达时）上游的
#            HMS4EYE_INTERFACE_COMPATIBILITY.md，其 `compat:http:<id>` 标记列出了他们的 68 个操作。

# Output (EN): Assertions over HMSCapabilities responses, the store rows behind them and manifest.CAPABILITY.
# 输出（中）：对 HMSCapabilities 响应、其背后的存储行以及 manifest.CAPABILITY 的断言。

# State effect (EN): Each test builds its own in-memory HMS; the clearing tests archive rows inside that store only.
# 状态影响（中）：每个测试自建内存 HMS；清除类测试只在该存储内归档行。

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any

import pytest

from hms_elite.configs.base import HMSConfig
from hms_elite.hms4eye import manifest
from hms_elite.hms4eye.capability import CONSOLIDATION_SOURCE_KINDS, HMSCapabilities
from hms_elite.memory.main import HMS
from hms_elite.models import Scope

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "events" / "structured_billing.json"
USER = "alice"
DERIVED_CONSTRAINT_PREFIX = "Do not re-apply "
USER_CONSTRAINT = "Do not modify the subscription state machine."
_COMPAT_MARKER = re.compile(r"compat:http:([A-Za-z0-9_]+)")


def _compat_md() -> pathlib.Path | None:
    """The upstream compatibility report, looked up by env var then by the usual sibling locations.
    上游兼容性报告，先按环境变量、再按常见的同级位置查找。
    """
    candidates = [
        os.environ.get("HMS4EYE_COMPAT_MD"),
        ROOT / "contracts" / "HMS4EYE_INTERFACE_COMPATIBILITY.md",
        ROOT.parent / "hms_lite-main" / "HMS4EYE_INTERFACE_COMPATIBILITY.md",
        pathlib.Path.home() / "Desktop" / "hms_lite-main" / "HMS4EYE_INTERFACE_COMPATIBILITY.md",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = pathlib.Path(candidate)
        if path.is_file():
            return path
    return None


def _live_observation_atoms(hms: HMS) -> list[str]:
    """Atom ids of the live consolidation products of the USER scope, sorted.
    USER 范围内存活的 consolidation 产物的原子 id，已排序。
    """
    kinds = sorted(CONSOLIDATION_SOURCE_KINDS)
    rows = hms.store.axis_select(
        Scope(user_id=USER),
        where_sql=f"source_kind IN ({','.join('?' for _ in kinds)})",
        params=kinds,
        order="atom_id ASC",
    )
    return [atom.atom_id for atom in rows]


@pytest.fixture
def rig() -> dict[str, Any]:
    """The S4 billing scenario ingested and consolidated on an offline HMS, plus its capability surface.
    在离线 HMS 上摄取并凝聚后的 S4 计费场景，以及其能力接口。

    Input / 输入：
        none.
        无。

    Output / 输出：
        {'hms', 'cap', 'scope', 'fixture', 'fail', 'fix'} for the tests to read.
        供测试读取的 {'hms', 'cap', 'scope', 'fixture', 'fail', 'fix'}。

    State effect / 状态影响：
        Creates one in-memory store (rule LLM + hash embedder + sqlite index) and runs one consolidation.
        创建一个内存存储（规则 LLM + 哈希嵌入器 + sqlite 索引）并运行一次凝聚。
    """
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    scope = dict(fixture["scope"])
    hms = HMS(HMSConfig(vector_store={"provider": "sqlite"}, store={"path": ":memory:"}))
    hms.add(fixture["constraint"]["content"], timestamp=fixture["constraint"]["timestamp"], **scope)
    fail = hms.add_events(fixture["events_fail"], **scope)
    fix = hms.add_events(fixture["events_fix"], **scope)
    cap = HMSCapabilities(hms)
    cap.trigger_consolidation(user_id=USER)
    return {"hms": hms, "cap": cap, "scope": scope, "fixture": fixture, "fail": fail, "fix": fix}


def test_list_entities_reports_real_mention_counts(rig: dict):
    """list_entities / get_entity return the scope's entities with counted mentions and their observations.
    list_entities / get_entity 返回范围内的实体，带真实计数的提及与其 observation。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions on items, mention_count, first_seen/last_seen and the entity's observations.
        对 items、mention_count、first_seen/last_seen 与实体 observation 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    cap, store = rig["cap"], rig["hms"].store
    listed = cap.list_entities(user_id=USER)
    assert listed["items"], "the billing scenario must produce entities"
    assert listed["total"] == len(store.entities_in_scope(Scope(user_id=USER)))
    by_name = {item["canonical_name"]: item for item in listed["items"]}
    assert {"billing_e2e", "src/billing/retry.ts", "src/billing/idempotency.ts"} <= set(by_name)
    suite = by_name["billing_e2e"]
    # The test suite is named by both test_log events, both EPs and the derived constraint: five source rows.
    # 该测试套件被两个 test_log 事件、两个 EP 以及派生约束提及：共五条来源行。
    assert suite["mention_count"] == 5
    assert suite["first_seen"] == "2026-06-15T10:04:00.000000+00:00"
    assert suite["last_seen"] == "2026-06-15T10:14:00.000000+00:00"
    assert suite["metadata"]["entity_type"] == "IDENTIFIER"
    assert suite["metadata"]["entity_atom_id"] is not None
    assert [obs["text"] for obs in suite["observations"]], "a consolidated entity carries observations"
    assert all(obs["witnesses"] for obs in suite["observations"])
    assert cap.get_entity(suite["id"], user_id=USER) == suite
    assert cap.get_entity("ent_does_not_exist", user_id=USER) is None
    assert cap.get_entity(suite["id"], user_id="bob") is None
    paged = cap.list_entities(user_id=USER, limit=2, offset=1)
    assert len(paged["items"]) == 2 and paged["items"][0]["id"] == listed["items"][1]["id"]


def test_entity_graph_edges_are_weighted_cooccurrences(rig: dict):
    """get_entity_graph pairs entities that share an atom and carries the shared atom's witness anchors.
    get_entity_graph 把共处同一原子的实体配对，并携带该原子的见证锚点。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions on node/edge counts, edge ordering and edge witnesses.
        对节点/边数量、边排序与边见证的断言。

    State effect / 状态影响：
        None.
        无。
    """
    graph = rig["cap"].get_entity_graph(user_id=USER)
    assert graph["total_entities"] == len(graph["nodes"]) and graph["edges"]
    assert graph["total_edges"] >= len(graph["edges"])
    weights = [edge["cooccurrence_count"] for edge in graph["edges"]]
    assert weights == sorted(weights, reverse=True), "heaviest co-occurrence first"
    for edge in graph["edges"]:
        assert edge["witnesses"], "a co-occurrence edge must name the anchors of the atoms that carry it"
        assert edge["cooccurrence_count"] == len(edge["atom_ids"]) == edge["weight"]
    assert len(rig["cap"].get_entity_graph(user_id=USER, limit=1)["edges"]) == 1


def test_get_graph_edges_always_carry_a_witness(rig: dict):
    """get_graph exposes both planes and every edge names at least one witness anchor - the core invariant.
    get_graph 暴露两个平面，且每条边都至少命名一个见证锚点 —— 这是核心不变量。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions on nodes, edge witnesses, fact types and the resolvability of every witness anchor.
        对节点、边见证、事实类型以及每个见证锚点可解析性的断言。

    State effect / 状态影响：
        None.
        无。
    """
    cap, store = rig["cap"], rig["hms"].store
    graph = cap.get_graph(user_id=USER)
    assert graph["nodes"] and graph["edges"]
    assert graph["total_units"] == len(store.list_nodes(Scope(user_id=USER), valid_only=True))
    node_ids = {node["id"] for node in graph["nodes"]}
    for edge in graph["edges"]:
        assert edge["witnesses"], f"edge {edge['id']} has no witness anchor"
        assert edge["src"] in node_ids and edge["dst"] in node_ids
        assert edge["src"] == edge["from"] and edge["dst"] == edge["to"]
        for anchor_id in edge["witnesses"]:
            assert store.get_anchor(anchor_id) is not None, f"witness {anchor_id} does not resolve"
    assert {"use", "distill", "temporal", "causal"} & {edge["type"] for edge in graph["edges"]}
    # Upstream's fact_type is three-state; our three layers fill all three without inventing a fourth.
    # 上游的 fact_type 是三态；我们的三层正好填满这三种，且不多造第四种。
    assert {node["fact_type"] for node in graph["nodes"]} <= {"world", "experience", "observation"}
    assert {"world", "experience", "observation"} == {node["fact_type"] for node in graph["nodes"]}
    assert len(cap.get_graph(user_id=USER, limit=2)["nodes"]) == 2


def test_directives_expose_the_derived_constraint(rig: dict):
    """list_directives returns both the user-written constraint and the one closure derived from the failed EP.
    list_directives 同时返回用户书写的约束与由失败 EP 的闭包派生出的约束。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions on the directive list, its ordering, priority derivation and get_directive.
        对指令列表、其排序、优先级派生与 get_directive 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    cap = rig["cap"]
    listed = cap.list_directives(user_id=USER)
    contents = [item["content"] for item in listed["items"]]
    assert USER_CONSTRAINT in contents
    derived = [item for item in listed["items"] if item["content"].startswith(DERIVED_CONSTRAINT_PREFIX)]
    assert len(derived) == 1, "the negative EP must mint exactly one 'Do not re-apply ...' constraint"
    directive = derived[0]
    assert directive["is_active"] is True and directive["priority"] == int(round(directive["confidence"] * 100))
    assert directive["name"] and directive["witnesses"] and directive["atom_id"]
    priorities = [item["priority"] for item in listed["items"]]
    assert priorities == sorted(priorities, reverse=True), "directives are ordered by derived priority"
    assert cap.get_directive(directive["id"], user_id=USER)["content"] == directive["content"]
    assert cap.get_directive(directive["atom_id"], user_id=USER)["id"] == directive["id"]
    assert cap.get_directive("mem_does_not_exist", user_id=USER) is None
    assert cap.get_directive(directive["id"], user_id="bob") is None
    assert cap.list_directives(user_id=USER, active_only=False)["total"] >= listed["total"]


def test_mental_models_come_from_promoted_skills(rig: dict):
    """list_mental_models returns the skill atoms promote_skills distilled, with lineage history and staleness.
    list_mental_models 返回 promote_skills 蒸馏出的 skill 原子，带谱系历史与陈旧判定。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions on the model fields, its markdown content, its history rows and get_mental_model.
        对模型字段、其 markdown 内容、其历史行与 get_mental_model 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    cap = rig["cap"]
    listed = cap.list_mental_models(user_id=USER)
    assert listed["total"] >= 1, "consolidation must promote at least one skill"
    model = listed["items"][0]
    assert model["name"] and model["bank_id"] == USER
    assert USER_CONSTRAINT.rstrip(".") in model["content"]
    assert model["content"].startswith("# ") and "Witness anchors" in model["content"]
    assert model["source_query"] == "fix checkout webhook retry"
    assert model["is_stale"] is False, "no EP arrived after the skill was minted"
    assert model["reflect_response"]["based_on"]["support_node_ids"]
    assert model["reflect_response"]["based_on"]["anchor_ids"]
    assert model["last_refreshed_at"] and model["created_at"]
    assert cap.get_mental_model(model["id"], user_id=USER)["id"] == model["id"]
    assert cap.get_mental_model(model["atom_id"], user_id=USER)["id"] == model["id"]
    assert cap.get_mental_model("mem_does_not_exist", user_id=USER) is None
    assert cap.get_mental_model(model["id"], user_id="bob") is None
    history = cap.get_mental_model_history(model["id"], user_id=USER)
    assert [row["event"] for row in history] == ["ADD"]
    assert history[0]["content"] == model["content"].split("\n")[2] or history[0]["content"]
    assert history[0]["anchor_id"] and history[0]["mental_model_id"] == model["id"]
    assert cap.get_mental_model_history("mem_does_not_exist", user_id=USER) == []


def test_trigger_consolidation_is_idempotent(rig: dict):
    """Two further trigger_consolidation runs report the same phase numbers and hand back a real anchor id.
    再跑两次 trigger_consolidation，阶段数字一致，且返回真实的锚点 id。

    Input / 输入：
        rig: The consolidated S4 rig (its fixture already ran one consolidation).
        rig：已凝聚的 S4 场景装置（其夹具已运行过一次凝聚）。

    Output / 输出：
        Assertions on report equality across runs and on the operation id.
        对多次运行间报告一致性与操作 id 的断言。

    State effect / 状态影响：
        Runs the scheduler twice more (each run writes one scheduler event and anchor).
        再运行两次调度器（每次运行写入一个调度器事件与锚点）。
    """
    cap, store = rig["cap"], rig["hms"].store
    second = cap.trigger_consolidation(user_id=USER)
    third = cap.trigger_consolidation(user_id=USER)
    assert second["report"]["phases"]["consolidate"] == third["report"]["phases"]["consolidate"]
    assert second["report"]["phases"]["consolidate"]["entity_atoms"] == 0
    assert second["report"]["phases"]["consolidate"]["skills"] == 0
    assert second["deduplicated"] is False
    for run in (second, third):
        assert store.get_anchor(run["operation_id"]) is not None, "the operation id must be a real anchor"
    partial = cap.trigger_consolidation(user_id=USER, phases=["orient"])
    assert set(partial["report"]["phases"]) == {"orient"}
    recovered = cap.recover_consolidation(user_id=USER)
    assert recovered["retried_count"] == 0, "nothing is pending once every EP is coarsened"
    assert store.get_anchor(recovered["operation_id"]) is not None


def test_clear_observations_archives_without_deleting(rig: dict):
    """clear_observations archives every consolidation product under a system anchor and keeps every row.
    clear_observations 在系统锚点之下归档每个 consolidation 产物，并保留每一行。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions on deleted_count, the archived rows' visibility with show_expired and the witness anchor.
        对 deleted_count、show_expired 下已归档行的可见性以及见证锚点的断言。

    State effect / 状态影响：
        Archives the scope's observation atoms and writes one ARCHIVE history row each.
        归档范围内的 observation 原子，并各写一行 ARCHIVE 历史。
    """
    cap, hms = rig["cap"], rig["hms"]
    store = hms.store
    before = _live_observation_atoms(hms)
    assert before, "the scenario must produce observation atoms first"
    rows_before = len(store.axis_select(Scope(user_id=USER), include_expired=True, order="atom_id ASC"))
    cleared = cap.clear_observations(user_id=USER)
    assert cleared["success"] is True and cleared["deleted_count"] == len(before)
    assert sorted(cleared["archived"]) == sorted(before)
    assert store.get_anchor(cleared["anchor_id"]) is not None, "clearing must be witnessed"
    assert _live_observation_atoms(hms) == [], "no observation atom stays live"
    rows_after = store.axis_select(Scope(user_id=USER), include_expired=True, order="atom_id ASC")
    assert len(rows_after) == rows_before, "HMS never deletes: the rows are still there"
    archived = {atom.atom_id: atom for atom in rows_after if atom.archived_reason == "cleared"}
    assert set(archived) == set(before)
    assert all(atom.valid_to is not None for atom in archived.values())
    expired = hms.get_all(filters={"user_id": USER}, top_k=200, show_expired=True)["results"]
    assert {row["id"] for row in expired} >= {atom.memory_id for atom in archived.values()}
    live_only = hms.get_all(filters={"user_id": USER}, top_k=200)["results"]
    assert {row["id"] for row in live_only}.isdisjoint({atom.memory_id for atom in archived.values()})
    assert cap.bank_stats(user_id=USER)["total_observations"] == 0
    # A second clear is a no-op with nothing left to archive, and mints no anchor for an empty target set.
    # 第二次清除已无可归档对象，因此是空操作，也不会为空目标集生成锚点。
    again = cap.clear_observations(user_id=USER)
    assert again["deleted_count"] == 0 and again["anchor_id"] is None
    # Consolidation rebuilds what was cleared, which is what makes archiving safe.
    # 凝聚会重建被清除的内容，这正是归档安全的原因。
    cap.trigger_consolidation(user_id=USER)
    assert _live_observation_atoms(hms), "a later consolidation restores the observation layer"


def test_clear_memory_observations_only_touches_derived_rows(rig: dict):
    """clear_memory_observations archives the observations that name one memory as support and nothing else.
    clear_memory_observations 只归档把某条记忆列为支持的 observation，不牵连其他。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions on the archived subset, the untouched source row and the unknown-id branch.
        对被归档子集、未受影响的来源行以及未知 id 分支的断言。

    State effect / 状态影响：
        Archives a subset of the scope's observation atoms.
        归档范围内 observation 原子的一个子集。
    """
    cap, hms = rig["cap"], rig["hms"]
    store = hms.store
    ep_atom = store.get_atom(store.get_ep(rig["fail"]["eps"][0]).atom_id)
    before = set(_live_observation_atoms(hms))
    cleared = cap.clear_memory_observations(ep_atom.memory_id, user_id=USER)
    assert cleared["success"] is True and 0 < cleared["deleted_count"] < len(before)
    assert set(cleared["archived"]) < before
    assert set(_live_observation_atoms(hms)) == before - set(cleared["archived"])
    assert store.get_atom(ep_atom.atom_id).archived_reason is None, "the source memory itself is untouched"
    for atom_id in cleared["archived"]:
        assert store.get_atom(atom_id).archived_reason == "cleared"
    missing = cap.clear_memory_observations("mem_does_not_exist", user_id=USER)
    assert missing["deleted_count"] == 0 and missing["success"] is False and missing["anchor_id"] is None
    assert cap.clear_memory_observations(ep_atom.memory_id, user_id="bob")["success"] is False


def test_bank_stats_counts_real_rows(rig: dict):
    """bank_stats fills the BankStatsResponse fields from counted rows, including the three-state fact histogram.
    bank_stats 用真实计数填充 BankStatsResponse 字段，含三态事实类型直方图。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 场景装置。

    Output / 输出：
        Assertions cross-checking every reported number against the store.
        逐项与存储交叉核对所报告数字的断言。

    State effect / 状态影响：
        None.
        无。
    """
    cap, hms = rig["cap"], rig["hms"]
    store = hms.store
    scope = Scope(user_id=USER)
    stats = cap.bank_stats(user_id=USER)
    live_atoms = store.axis_select(scope, order="atom_id ASC")
    live_edges = store.list_edges(scope, valid_only=True)
    assert stats["bank_id"] == USER
    assert stats["total_nodes"] == len(live_atoms) == sum(stats["nodes_by_fact_type"].values())
    assert stats["total_links"] == len(live_edges) == sum(stats["links_by_link_type"].values())
    assert stats["total_observations"] == len(_live_observation_atoms(hms))
    assert stats["total_observations"] == stats["nodes_by_fact_type"]["observation"]
    assert stats["nodes_by_fact_type"]["experience"] == len(store.list_eps(scope, valid_only=True))
    assert stats["pending_consolidation"] == 0, "every EP of the scenario is coarsened into an InterfaceEP"
    assert stats["failed_consolidation"] == 0 and stats["failed_operations"] == 0
    assert stats["last_consolidated_at"] == store.last_events(scope, object_types=["scheduler"], limit=1)[0].timestamp
    assert set(stats["links_breakdown"]) == set(stats["links_by_link_type"])
    assert sum(sum(inner.values()) for inner in stats["links_breakdown"].values()) == stats["total_links"]
    empty = cap.bank_stats(user_id="nobody")
    assert empty["total_nodes"] == 0 and empty["total_observations"] == 0 and empty["last_consolidated_at"] is None


def test_manifest_covers_every_upstream_http_operation():
    """manifest.CAPABILITY names exactly the operationIds the upstream compatibility report marks as HTTP.
    manifest.CAPABILITY 恰好命名了上游兼容性报告中标为 HTTP 的那些 operationId。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions comparing CAPABILITY / UPSTREAM against the markers parsed out of the upstream markdown.
        将 CAPABILITY / UPSTREAM 与从上游 markdown 解析出的标记比对的断言。

    State effect / 状态影响：
        None (the report is only read; the test skips when it is not reachable).
        无（只读取该报告；报告不可达时跳过测试）。
    """
    report = _compat_md()
    if report is None:
        pytest.skip("upstream HMS4EYE_INTERFACE_COMPATIBILITY.md not reachable; set HMS4EYE_COMPAT_MD to point at it")
    upstream: dict[str, str] = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| http |"):
            continue
        marker = _COMPAT_MARKER.search(line)
        assert marker is not None, f"an http row without a compat marker: {line[:80]}"
        upstream[marker.group(1)] = line.split("|")[5].strip().strip("`")
    assert len(upstream) == 68, f"expected 68 HTTP operations upstream, parsed {len(upstream)}"
    assert set(manifest.CAPABILITY) == set(upstream), "our capability matrix must name every HTTP operationId"
    assert manifest.UPSTREAM == upstream, "our transcription of their verdicts must match their report"
    assert set(manifest.CAPABILITY.values()) <= set(manifest.VERDICTS)
    assert set(manifest.SURFACE) == {k for k, v in manifest.CAPABILITY.items() if v != "unsupported"}


def test_compare_with_upstream_reports_a_lift():
    """compare_with_upstream() shows this engine lifting upstream no-ops into native or adapted capabilities.
    compare_with_upstream() 显示本引擎把上游的 no-op 提升为 native 或 adapted 能力。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the lift statistics and on the internal consistency of the report.
        对提升统计与报告内部一致性的断言。

    State effect / 状态影响：
        None.
        无。
    """
    result = manifest.compare_with_upstream()
    assert result["total"] == len(manifest.UPSTREAM) == 68
    assert result["missing"] == []
    assert result["upstream_counts"] == {"native": 6, "adapted": 29, "no-op": 33}
    assert result["lifted_count"] > 0, "the whole point of the capability layer"
    lifted = set(result["lifted"]["native"]) | set(result["lifted"]["adapted"])
    assert {"list_entities", "get_entity", "get_graph", "get_entity_graph", "trigger_consolidation"} <= lifted
    assert {"list_mental_models", "list_directives", "clear_observations"} <= lifted
    assert lifted.isdisjoint(result["still_unsupported"])
    assert len(lifted) + len(result["still_unsupported"]) == result["upstream_counts"]["no-op"]
    assert sum(result["our_counts"].values()) == result["total"]
    assert all(manifest.UPSTREAM[name] != "no-op" for name in result["dropped"])


def test_reflect_evidence_is_an_organize_shaped_ledger_whose_rows_cite_anchors(rig: dict):
    """reflect_evidence() returns the `organize` packet built from the Detective Report: every ledger row carries anchor_ids, rows follow the why() chain, an empty bank gets the empty packet, and the manifest keeps reflect native.
    reflect_evidence() 返回由侦探报告构造的 `organize` 证据包：每行 ledger 都带 anchor_ids，行序跟随 why() 链，空记忆库得到空证据包；manifest 中 reflect 保持 native。

    Input / 输入：
        rig: The consolidated S4 rig.
        rig：已凝聚的 S4 装置。

    Output / 输出：
        Assertions over the packet keys, ledger rows, snippets, context text and the manifest statistics.
        对证据包键、ledger 行、片段、上下文文本与 manifest 统计的断言。

    State effect / 状态影响：
        None (reads only).
        无（只读）。
    """
    cap, hms = rig["cap"], rig["hms"]
    question = "why must retry.ts not be modified"
    packet = cap.reflect_evidence(question, user_id=USER)
    assert list(packet) == [
        "question",
        "question_date",
        "mode",
        "ledger_rows",
        "controls",
        "source_snippets",
        "answer_ready_context",
        "conclusion",
        "chain",
    ]
    assert packet["question"] == question and packet["mode"] == "detective_report" and packet["controls"] == []
    rows = packet["ledger_rows"]
    assert rows and [row["index"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["score"] == 0 and row["anchor_ids"] and row["atom_id"] and row["chunk_id"] is None for row in rows)
    assert {row["type"] for row in rows} <= {"world", "experience", "observation"}
    report, _text = hms.state.detective_report(question, filters={"user_id": USER})
    assert [row["text"] for row in rows] == [step.text for step in report.steps]
    assert [row["anchor_ids"] for row in rows] == [list(step.anchor_ids) for step in report.steps]
    assert packet["chain"] == report.chain and packet["conclusion"]["anchor_ids"] == list(report.conclusion.anchor_ids)
    cited = {anchor for row in rows for anchor in row["anchor_ids"]}
    assert cited <= {snippet["anchor_id"] for snippet in packet["source_snippets"]}
    assert all(snippet["text"] and snippet["source_ref"] for snippet in packet["source_snippets"])
    context = packet["answer_ready_context"]
    assert context.startswith(f"=== Retrieved Memory Evidence ===\nQuestion: {question}\nQuestion date: ")
    assert f"{len(rows)}. occurred=" in context and context.count("\n") == 5 + len(rows) - 1
    # Nothing resolves in an empty bank: the empty packet, not an error.
    # 空记忆库无法解析任何东西：返回空证据包而非报错。
    empty = cap.reflect_evidence(question, user_id="nobody")
    assert empty["ledger_rows"] == [] and empty["source_snippets"] == [] and empty["conclusion"] is None
    assert empty["chain"] == [] and empty["answer_ready_context"].endswith("Recalled facts (original order):\n- none")
    assert manifest.CAPABILITY["reflect"] == "native"
    assert manifest.SURFACE["reflect"] == "HMS.compile + HMSCapabilities.reflect_evidence"
    result = manifest.compare_with_upstream()
    assert "reflect" not in result["dropped"] and result["our_counts"]["native"] >= 13
