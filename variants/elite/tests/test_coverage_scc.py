"""Tests for compiler/coverage.py and compiler/scc.py: requirement extraction, greedy cover, layered compile, lowering.
compiler/coverage.py 与 compiler/scc.py 的测试：需求抽取、贪心覆盖、分层编译、降低。
"""

# Purpose (EN): Pin the HMS §5.1 coverage / SCC contract on hand-built ContextIRs and on the S4 structured scenario,
#               and its wiring into compile() (selection='coverage' / compile(mode='coverage')).
# 目的（中）：在手工构造的 ContextIR 与 S4 结构化场景上钉住 HMS §5.1 的覆盖 / SCC 契约，及其接入 compile()
#            （selection='coverage' / compile(mode='coverage')）。

# Input (EN): Hand-built ContextItems / ContextIRs, tests/fixtures/events/structured_billing.json, conftest fixtures.
# 输入（中）：手工构造的 ContextItem / ContextIR、tests/fixtures/events/structured_billing.json、conftest 夹具。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): None outside the in-memory SQLite store of the `hms` fixture.
# 状态影响（中）：除 `hms` 夹具的内存 SQLite 存储外无副作用。

from __future__ import annotations

import random
from typing import Any

import pytest

from hms_elite.compiler.coverage import (
    KINDS,
    Requirement,
    coverage_matrix,
    covers,
    extract_requirements,
    select_for_coverage,
)
from hms_elite.compiler.scc import (
    DEFAULT_LAYERS,
    REQUIREMENTS_COVERED,
    HierarchicalIR,
    HierarchicalPCIR,
    Layer,
    compile_hierarchical,
    layer_of,
    lower,
)
from hms_elite.enums import Scale
from hms_elite.exceptions import ProofObligationError, ValidationError
from hms_elite.memory.utils import estimate_tokens
from hms_elite.models import ContextIR, ContextItem, Scope
from hms_elite.mvmi.planner import plan_query

AS_OF = "2026-06-15T12:00:00Z"
S4_QUERY = "fix checkout webhook retry"
S4_WHY = "why must retry.ts not be modified"
S4_DRILL = "why did billing_e2e fail with subscription_state_regression"
S4_CONSTRAINT = "Do not modify the subscription state machine."


def _item(
    item_id: str,
    slot: str,
    text: str,
    *,
    scale: str | None = None,
    score: float = 0.0,
    anchors: list[str] | None = None,
    time_anchor: str | None = None,
    explain: dict[str, Any] | None = None,
    tokens: int | None = None,
) -> ContextItem:
    """A hand-built ContextItem (tokens default to estimate_tokens(text)).
    手工构造的 ContextItem（tokens 默认为 estimate_tokens(text)）。
    """
    return ContextItem(
        item_id=item_id,
        slot=slot,
        text=text,
        atom_id=None if item_id.startswith("anc:") else item_id,
        score=score,
        scale_level=scale,
        time_anchor=time_anchor,
        source="interface_ep" if scale in ("L2_meso",) else ("system" if item_id.startswith("anc:") else "axis"),
        tokens=estimate_tokens(text) if tokens is None else tokens,
        explain=explain or {},
        anchor_ids=anchors or ["anc_a"],
    )


def _evidence(anchor_id: str) -> ContextItem:
    """An 'anc:<id>' evidence row.
    一条 'anc:<id>' 证据行。
    """
    return _item(f"anc:{anchor_id}", "audit", f"file://{anchor_id} sha256:00 @{AS_OF}", anchors=[anchor_id])


def _req(req_id: str, kind: str, text: str, tokens: tuple[str, ...], *, weight: float = 1.0) -> Requirement:
    """A hand-built Requirement.
    手工构造的 Requirement。
    """
    return Requirement(req_id=req_id, kind=kind, text=text, weight=weight, tokens=tokens)


def _ir(query: str, items: list[ContextItem], *, anchors: list[str], conflicts: list[str] | None = None) -> ContextIR:
    """A hand-built ContextIR whose retrieval['anchors'] carries rows for `anchors`.
    手工构造的 ContextIR，其 retrieval['anchors'] 携带 `anchors` 的行。
    """
    slots: dict[str, list[ContextItem]] = {
        n: [] for n in ("facts", "episodes", "preferences", "constraints", "skills", "audit")
    }
    for item in items:
        slots[str(item.slot)].append(item)
    return ContextIR(
        query=query,
        goal=query,
        as_of=AS_OF,
        scope=Scope(user_id="alice", run_id="run-billing"),
        budget_tokens=1200,
        used_tokens=sum(i.tokens for i in items),
        retrieval={
            "plan": plan_query(query, as_of=AS_OF).as_dict(),
            "anchors": [{"anchor_id": a, "permission_scope": "private"} for a in anchors],
            "audience": None,
        },
        unresolved_conflicts=list(conflicts or []),
        **slots,
    )


# ---------------------------------------------------------------------------
# coverage.py — requirement extraction
# coverage.py —— 需求抽取
# ---------------------------------------------------------------------------
def test_extract_requirements_golden_queries():
    """The two golden queries and their ZH / temporal / entity siblings yield the expected requirement lists.
    两个黄金查询及其中文 / 时间 / 实体变体得到预期的需求列表。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over req_id, kind, weight, tokens and window.
        对 req_id、kind、weight、tokens 与 window 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    fix = extract_requirements(S4_QUERY, as_of=AS_OF)
    assert [(r.req_id, r.kind, r.weight) for r in fix] == [("goal:fix checkout webhook retry", "goal", 1.0)]
    assert fix[0].tokens == ("fix", "checkout", "webhook", "retry") and fix[0].window is None

    why = extract_requirements(S4_WHY, as_of=AS_OF)
    assert [(r.req_id, r.kind) for r in why] == [
        ("constraint:retry.ts not be modified", "constraint"),
        ("identifier:retry.ts", "identifier"),
    ]
    assert "retry.ts" in why[0].tokens and why[0].weight == 0.9 and why[1].tokens == ("retry.ts",)
    assert set(r.kind for r in why) <= set(KINDS)

    # A plan dict (ContextIR.retrieval['plan']) or a QueryPlan gives the same answer as planning from as_of.
    # 规划字典（ContextIR.retrieval['plan']）或 QueryPlan 与按 as_of 规划得到相同答案。
    plan = plan_query(S4_WHY, as_of=AS_OF)
    assert extract_requirements(S4_WHY, plan) == why == extract_requirements(S4_WHY, plan.as_dict())

    sarah = extract_requirements("what happened with Sarah last week", as_of=AS_OF)
    assert [r.kind for r in sarah] == ["entity", "time"]
    assert sarah[0].req_id == "entity:sarah" and sarah[0].weight == 0.8
    assert sarah[1].window == ("2026-06-08T00:00:00.000000+00:00", "2026-06-14T23:59:59.999999+00:00")
    assert sarah[1].req_id == f"time:{sarah[1].window[0]}..{sarah[1].window[1]}" and sarah[1].weight == 0.6

    zh = extract_requirements("修复 checkout webhook 重试，不要改 retry.ts", as_of=AS_OF)
    assert [r.kind for r in zh] == ["goal", "constraint", "identifier"]
    assert (
        zh[0].text == "修复checkout webhook 重试"
        and "retry.ts" in zh[1].tokens
        and zh[2].req_id == "identifier:retry.ts"
    )

    # No cue at all → the whole query becomes one goal so coverage is never vacuous; empty → nothing.
    # 完全没有线索 → 整个查询成为一个 goal，使覆盖不至于空洞；空查询 → 无需求。
    bare = extract_requirements("coffee preferences overview", as_of=AS_OF)
    assert [r.kind for r in bare] == ["goal"] and bare[0].tokens == ("coffee", "preference", "overview")
    assert extract_requirements("", as_of=AS_OF) == []
    with pytest.raises(TypeError):
        extract_requirements(S4_QUERY, plan=42)
    assert [r.as_dict()["req_id"] for r in why] == [r.req_id for r in why]


def test_covers_rules():
    """identifier = containment (basename too), entity = normalized containment, time = window, text = token share.
    identifier = 包含（含基名），entity = 归一化包含，time = 时间窗，文本类 = 词元占比。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Boolean assertions per requirement kind.
        每种需求种类的布尔断言。

    State effect / 状态影响：
        None.
        无。
    """
    ident = _req("identifier:retry.ts", "identifier", "retry.ts", ("retry.ts",))
    assert covers(_item("a", "facts", "patched src/billing/retry.ts"), ident)
    assert not covers(_item("b", "facts", "patched idempotency.ts"), ident)
    path = _req("identifier:src/billing/retry.ts", "identifier", "src/billing/retry.ts", ("src/billing/retry.ts",))
    assert covers(_item("c", "facts", "retry.ts changed"), path)

    entity = _req("entity:sarah", "entity", "Sarah", ("sarah",), weight=0.8)
    assert covers(_item("d", "facts", "User met SARAH in Berlin."), entity)
    assert not covers(_item("e", "facts", "User met Sam."), entity)

    window = Requirement(
        req_id="time:w",
        kind="time",
        text="last week",
        weight=0.6,
        window=("2026-06-08T00:00:00Z", "2026-06-14T23:59:59Z"),
    )
    assert covers(_item("f", "facts", "x", time_anchor="2026-06-10T09:00:00Z"), window)
    assert not covers(_item("g", "facts", "x", time_anchor="2026-06-15T09:00:00Z"), window)
    assert not covers(_item("h", "facts", "x"), window)

    goal = _req("goal:fix checkout webhook retry", "goal", S4_QUERY, ("fix", "checkout", "webhook", "retry"))
    assert covers(_item("i", "episodes", "[negative] fix checkout webhook retry → patch retry.ts"), goal)
    # 2 of 4 tokens → covered; 1 of 4 → not covered.
    # 4 个词元命中 2 个 → 覆盖；命中 1 个 → 不覆盖。
    assert covers(_item("j", "facts", "the checkout webhook is flaky"), goal)
    assert not covers(_item("k", "facts", "the webhook"), goal)
    constraint = _req("constraint:x", "constraint", "retry.ts not be modified", ("retry.ts", "modifi"), weight=0.9)
    assert covers(_item("l", "constraints", "Do not re-apply 'patch src/billing/retry.ts'"), constraint)
    assert not covers(_item("m", "facts", "x"), _req("goal:empty", "goal", "", ()))
    matrix = coverage_matrix(
        [_item("a", "facts", "patched src/billing/retry.ts"), _item("m", "facts", "x")], [ident, goal]
    )
    assert matrix == {"a": ["identifier:retry.ts"], "m": []}


def test_select_for_coverage_greedy_deterministic_and_budget():
    """Greedy cover prefers most-newly-covered then weight per token, ties by item_id; budget, floors and evidence hold.
    贪心覆盖优先新增覆盖最多、再看单位 token 权重，并列按 item_id；预算、保底与证据规则成立。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over selected ids, covered map, uncovered list and overlap_stats.
        对选中 id、覆盖映射、未覆盖列表与 overlap_stats 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    r1 = _req("goal:g", "goal", "fix retry", ("fix", "retry"))
    r2 = _req("identifier:retry.ts", "identifier", "retry.ts", ("retry.ts",))
    r3 = _req("entity:sarah", "entity", "Sarah", ("sarah",), weight=0.8)
    both = _item("atm_both", "episodes", "fix retry via retry.ts", tokens=30)
    only_goal = _item("atm_goal", "facts", "fix retry now", tokens=10)
    only_id = _item("atm_id", "constraints", "never touch retry.ts", tokens=10)
    sarah = _item("atm_sarah", "facts", "Sarah owns billing", tokens=50)
    noise = _item("atm_noise", "facts", "unrelated", tokens=5, score=9.0)
    evidence = _evidence("anc_a")
    items = [noise, sarah, only_id, only_goal, both, evidence]

    tight = select_for_coverage(items, [r1, r2, r3], budget_tokens=45)
    assert tight.selected_ids == ["anc:anc_a", "atm_both"]
    assert tight.covered == {"goal:g": ["atm_both"], "identifier:retry.ts": ["atm_both"]}
    assert [r.req_id for r in tight.uncovered] == ["entity:sarah"]
    stats = tight.overlap_stats
    assert stats["requirements"] == 3 and stats["covered"] == 2 and stats["coverage_ratio"] == round(2 / 3, 4)
    assert stats["skipped_over_budget"] == ["atm_sarah"] and stats["rounds"] == 1 and stats["evidence"] == 1
    assert stats["used_tokens"] == 30 + evidence.tokens and stats["over_budget_evidence"] > 0
    assert tight.as_dict()["uncovered"][0]["req_id"] == "entity:sarah"

    wide = select_for_coverage(items, [r1, r2, r3], budget_tokens=500)
    assert wide.selected_ids == ["anc:anc_a", "atm_both", "atm_sarah"] and wide.uncovered == []
    assert wide.overlap_stats["redundancy"] == 1.0 and wide.overlap_stats["coverage_ratio"] == 1.0

    # Determinism: any input order gives the same selection.
    # 确定性：任意输入顺序得到相同的选择。
    for seed in range(5):
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        assert select_for_coverage(shuffled, [r1, r2, r3], budget_tokens=500).selected_ids == wide.selected_ids

    # Per-token gain: two single-requirement items → the cheaper one; equal cost → the smaller item_id.
    # 单位 token 收益：两个各覆盖一条需求的项 → 取更便宜者；同价 → 取较小的 item_id。
    cheap = select_for_coverage([sarah, _item("atm_z", "facts", "Sarah again", tokens=8)], [r3], budget_tokens=500)
    assert cheap.selected_ids == ["atm_z"]
    tie = select_for_coverage(
        [_item("atm_b", "facts", "Sarah b", tokens=8), _item("atm_a", "facts", "Sarah a", tokens=8)],
        [r3],
        budget_tokens=500,
    )
    assert tie.selected_ids == ["atm_a"]

    # Slot floors: after coverage, min_per_slot fills facts / constraints with the best-scored leftovers.
    # 槽位保底：覆盖之后，min_per_slot 用分数最高的剩余项填充 facts / constraints。
    floors = select_for_coverage(items, [r2], budget_tokens=500, min_per_slot=1)
    assert floors.selected_ids == ["anc:anc_a", "atm_id", "atm_noise", "atm_both"]
    assert floors.overlap_stats["floors_added"] == 2 and floors.covered == {
        "identifier:retry.ts": ["atm_id", "atm_both"]
    }
    assert floors.overlap_stats["redundancy"] == 2.0
    with pytest.raises(ValueError):
        select_for_coverage(items, [r1], budget_tokens=0)
    empty = select_for_coverage(items, [], budget_tokens=100)
    assert empty.selected_ids == ["anc:anc_a"] and empty.overlap_stats["coverage_ratio"] == 1.0


# ---------------------------------------------------------------------------
# scc.py — hierarchical compile and lowering on hand-built IRs
# scc.py —— 手工 IR 上的分层编译与降低
# ---------------------------------------------------------------------------
def _layered_ir(query: str = S4_DRILL) -> ContextIR:
    """A ContextIR with one L3 skill, one L2 episode + L2 invariant, two L1 episodes, one L1 constraint, evidence rows.
    含一个 L3 技能、一个 L2 情节 + L2 不变量、两个 L1 情节、一个 L1 约束与证据行的 ContextIR。
    """
    items = [
        _item("atm_skill", "skills", f"Skill: {S4_CONSTRAINT}", scale="L3_macro", anchors=["anc_uc"]),
        _item(
            "atm_iep",
            "episodes",
            "[positive] fix checkout webhook retry → patch src/billing/retry.ts → billing_e2e passed",
            scale="L2_meso",
            anchors=["anc_e1", "anc_e3"],
            explain={"kappa": [S4_CONSTRAINT]},
        ),
        _item(
            "atm_iep#inv:1",
            "audit",
            "do not: patch src/billing/retry.ts: add exponential backoff",
            scale="L2_meso",
            anchors=["anc_e1", "anc_e2"],
            explain={"reason": "invariant:failed_path"},
        ),
        _item(
            "atm_ep1",
            "episodes",
            "[negative] fix checkout webhook retry → patch src/billing/retry.ts → billing_e2e failed: "
            "subscription_state_regression",
            scale="L1_micro",
            anchors=["anc_e1", "anc_e2"],
            explain={"kappa": [S4_CONSTRAINT], "r": "negative"},
        ),
        _item(
            "atm_ep2",
            "episodes",
            "[positive] patch idempotency.ts → billing_e2e passed",
            scale="L1_micro",
            anchors=["anc_e3"],
        ),
        _item("atm_uc", "constraints", S4_CONSTRAINT, scale="state", anchors=["anc_uc"]),
        _item("atm_fact", "facts", "User lives in Berlin.", scale="state", anchors=["anc_f"]),
        _evidence("anc_uc"),
        _evidence("anc_e1"),
        _evidence("anc_e2"),
        _evidence("anc_e3"),
        _evidence("anc_f"),
    ]
    return _ir(query, items, anchors=["anc_uc", "anc_e1", "anc_e2", "anc_e3", "anc_f"])


def test_compile_hierarchical_l2_first_then_drill_to_l1():
    """L3/L2 are offered first; only the requirement they leave uncovered drills to L1; κ closure and evidence follow.
    先提供 L3/L2；只有它们留下的未覆盖需求下钻到 L1；κ 闭包与证据随之保留。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over LayerReports, the reduced ContextIR and retrieval['hierarchy'].
        对 LayerReport、缩减后的 ContextIR 与 retrieval['hierarchy'] 的断言。

    State effect / 状态影响：
        None.
        无。
    """
    ir = _layered_ir()
    assert [layer_of(i).value for i in ir.items()][:4] == ["L1_micro", "L2_meso", "L1_micro", "L1_micro"]
    hir = compile_hierarchical(ir)
    assert isinstance(hir, HierarchicalIR) and hir.budget_tokens == 1200
    assert [r.req_id for r in hir.requirements] == [
        "identifier:billing_e2e",
        "identifier:subscription_state_regression",
    ]
    l3, l2, l1 = hir.layers
    assert [r.layer for r in hir.layers] == [layer.value for layer in DEFAULT_LAYERS]
    assert l3.candidates == ["atm_skill"] and l3.selected == [] and l3.refined is True
    assert l2.candidates == ["atm_iep", "atm_iep#inv:1"] and l2.selected == ["atm_iep"]
    assert l2.covered == {"identifier:billing_e2e": ["atm_iep"]} and l2.uncovered == [
        "identifier:subscription_state_regression"
    ]
    assert l1.selected == ["atm_ep1"] and l1.covered == {"identifier:subscription_state_regression": ["atm_ep1"]}
    assert l1.uncovered == [] and l1.refined is True and hir.uncovered == []
    assert hir.kappa_closure == ["atm_uc"]
    reduced = hir.ir
    assert [i.item_id for i in reduced.episodes] == ["atm_iep", "atm_ep1"]
    assert [i.item_id for i in reduced.constraints] == ["atm_uc"] and reduced.facts == [] and reduced.skills == []
    assert [i.item_id for i in reduced.audit] == ["anc:anc_uc", "anc:anc_e1", "anc:anc_e2", "anc:anc_e3"]
    assert reduced.used_tokens == hir.used_tokens == sum(i.tokens for i in reduced.items())
    assert reduced.retrieval["hierarchy"] == hir.report() and reduced.retrieval["plan"] == ir.retrieval["plan"]
    assert reduced.query == ir.query and reduced.scope == ir.scope and "## Failed Paths\n- " in hir.to_text()

    # Everything covered at L2 → L1 is not refined and no L1 atom survives (the interface answers the question).
    # 全部在 L2 覆盖 → L1 不下钻，也没有 L1 原子留下（接口本身回答了问题）。
    why = compile_hierarchical(_layered_ir(S4_WHY))
    assert [(r.layer, r.selected, r.refined) for r in why.layers] == [
        ("L3_macro", [], True),
        ("L2_meso", ["atm_iep#inv:1"], True),
        ("L1_micro", [], False),
    ]
    assert why.covered == {
        "constraint:retry.ts not be modified": ["atm_iep#inv:1"],
        "identifier:retry.ts": ["atm_iep#inv:1"],
    }
    assert [i.item_id for i in why.ir.items()] == ["atm_iep#inv:1", "anc:anc_e1", "anc:anc_e2"]

    # Explicit requirements, layer order as strings, and a budget that cannot afford the L2 episode.
    # 显式需求、字符串形式的层次顺序，以及付不起 L2 情节的预算。
    goal = extract_requirements(S4_QUERY, as_of=AS_OF)
    bottom_up = compile_hierarchical(ir, requirements=goal, layers=["L1_micro", "L2_meso"])
    assert [r.layer for r in bottom_up.layers] == ["L1_micro", "L2_meso"]
    assert bottom_up.layers[0].selected == ["atm_ep1"] and bottom_up.layers[1].refined is False
    tight = compile_hierarchical(ir, requirements=goal, budget_tokens=25)
    assert tight.uncovered == goal and tight.ir.items() == [] and tight.ir.budget_tokens == 25
    with pytest.raises(ValueError):
        compile_hierarchical(ir, layers=[])
    with pytest.raises(ValueError):
        compile_hierarchical(ir, layers=["L9"])
    assert compile_hierarchical(ir).report() == hir.report()


def test_lower_obligations_conflict_and_uncovered():
    """lower() keeps the four base obligations and adds requirements_covered; conflicts / uncovered flip ok() to False.
    lower() 保留四条基础义务并新增 requirements_covered；冲突 / 未覆盖使 ok() 变为 False。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions over HierarchicalPCIR obligations, payload and ProofObligationError details.
        对 HierarchicalPCIR 义务、载荷与 ProofObligationError 细节的断言。

    State effect / 状态影响：
        None.
        无。
    """
    clean = lower(compile_hierarchical(_layered_ir()))
    assert isinstance(clean, HierarchicalPCIR) and clean.ok() is True
    names = [o["name"] for o in clean.all_obligations()]
    assert names == ["evidence_coverage", "no_unresolved_conflict", "governance", "not_revoked", REQUIREMENTS_COVERED]
    assert clean.all_obligations()[-1]["status"] == "satisfied"
    assert clean.payload["requirements"]["uncovered"] == [] and set(clean.payload["requirements"]["covered"]) == {
        "identifier:billing_e2e",
        "identifier:subscription_state_regression",
    }
    assert clean.witness["items"]["atm_ep1"] == ["anc_e1", "anc_e2"]
    assert clean.human_gloss.startswith("# Work State Card")

    # Uncovered requirement → requirements_covered violated, listed in details; require_witness raises.
    # 未覆盖需求 → requirements_covered 为 violated 并列在 details 中；require_witness 抛错。
    missing = _req("identifier:nowhere.ts", "identifier", "nowhere.ts", ("nowhere.ts",))
    hir = compile_hierarchical(_layered_ir(), requirements=[missing])
    pcir = lower(hir)
    assert pcir.ok() is False and pcir.coverage_obligations[0].status == "violated"
    assert [r["req_id"] for r in pcir.coverage_obligations[0].details["uncovered"]] == ["identifier:nowhere.ts"]
    assert pcir.payload["requirements"]["uncovered"] == ["identifier:nowhere.ts"]
    assert all(o.status != "violated" for o in pcir.proof_obligations)
    with pytest.raises(ProofObligationError) as excinfo:
        lower(hir, require_witness=True)
    assert excinfo.value.details["violated"] == [REQUIREMENTS_COVERED]
    assert excinfo.value.details["uncovered_requirements"] == ["identifier:nowhere.ts"]

    # Unresolved conflict → no_unresolved_conflict violated (base obligation) even though coverage is satisfied.
    # 未解决冲突 → no_unresolved_conflict 为 violated（基础义务），即便覆盖已满足。
    conflicted = _layered_ir().model_copy(update={"unresolved_conflicts": ["edge_conflict"]})
    pcir2 = lower(compile_hierarchical(conflicted))
    assert pcir2.ok() is False and pcir2.coverage_obligations[0].status == "satisfied"
    by_name = {o["name"]: o for o in pcir2.all_obligations()}
    assert by_name["no_unresolved_conflict"]["status"] == "violated"
    assert by_name["no_unresolved_conflict"]["details"]["unresolved_conflicts"] == ["edge_conflict"]
    with pytest.raises(ProofObligationError) as excinfo2:
        lower(compile_hierarchical(conflicted), require_witness=True)
    assert excinfo2.value.details["violated"] == ["no_unresolved_conflict"]

    # No requirements at all → the obligation is skipped, not violated.
    # 完全没有需求 → 义务为 skipped 而非 violated。
    none = lower(compile_hierarchical(_layered_ir(), requirements=[]))
    assert none.coverage_obligations[0].status == "skipped" and none.ok() is True


# ---------------------------------------------------------------------------
# S4 structured scenario through the public facade
# 通过公开门面的 S4 结构化场景
# ---------------------------------------------------------------------------
def _ingest_s4(hms, clock, events, scope4: dict[str, str]) -> None:
    """Ingest constraint + events_fail + events_fix and consolidate (L2 InterfaceEP + L3 skill appear).
    摄取约束 + events_fail + events_fix 并固化（出现 L2 InterfaceEP 与 L3 技能）。
    """
    fixture = events("structured_billing")
    clock.tick(seconds=60)
    hms.add(fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
    clock.tick(seconds=60)
    hms.add_events(fixture.events_fail, **scope4)
    clock.tick(seconds=60)
    hms.add_events(fixture.events_fix, **scope4)
    clock.tick(seconds=60)
    hms.consolidate(filters=scope4)


def test_scenario_s4_interface_first_then_refine(hms, clock, events, scope4):
    """On S4 after consolidate: L2 InterfaceEP items cover first, L1 atoms are refined only for what L2 left uncovered.
    在固化后的 S4 上：L2 InterfaceEP 项先覆盖，L1 原子仅为 L2 留下的未覆盖需求而下钻。

    Input / 输入：
        hms: The HMS fixture (rule LLM + hash embedder + parametrized index).
        hms：HMS 夹具（规则 LLM + 哈希嵌入器 + 参数化索引）。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

    Output / 输出：
        Assertions over the three queries' layer reports, reduced IRs and PC-IR obligations.
        对三个查询的层报告、缩减 IR 与 PC-IR 义务的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    _ingest_s4(hms, clock, events, scope4)
    stats = hms.graph_stats(filters=scope4)
    assert stats["interface_eps"] == 1 and stats["nodes"]["skill"] == 1

    clock.tick(seconds=60)
    deep = hms.compile(S4_QUERY, filters=scope4, mode="deep")
    layers = {layer_of(i).value for i in deep.items()}
    assert layers == {Scale.L1_micro.value, Scale.L2_meso.value, Scale.L3_macro.value}
    hir = compile_hierarchical(deep)
    assert hir.requirements[0].req_id == "goal:fix checkout webhook retry" and hir.uncovered == []
    l3, l2, l1 = hir.layers
    assert l3.selected == [] and len(l2.selected) == 1 and l1.selected == [] and l1.refined is False
    iep_item = next(i for i in deep.episodes if str(i.scale_level) == "L2_meso")
    assert l2.selected == [iep_item.item_id] and hir.covered == {hir.requirements[0].req_id: [iep_item.item_id]}
    # κ closure keeps both constraints although neither covers the goal.
    # κ 闭包保留两条约束，尽管它们都不覆盖目标。
    assert [i.text for i in hir.ir.constraints] == [c.text for c in deep.constraints]
    assert hir.kappa_closure == [c.item_id for c in deep.constraints]
    assert [i.item_id for i in hir.ir.episodes] == [iep_item.item_id] and hir.ir.skills == []
    assert hir.ir.used_tokens < deep.used_tokens and lower(hir).ok() is True
    assert set(hir.ir.anchor_ids()) <= set(deep.anchor_ids())

    clock.tick(seconds=60)
    why = compile_hierarchical(hms.compile(S4_WHY, filters=scope4, mode="deep"))
    assert [r.req_id for r in why.requirements] == ["constraint:retry.ts not be modified", "identifier:retry.ts"]
    assert why.layers[1].selected == [f"{iep_item.item_id}#inv:1"] and why.layers[2].refined is False
    assert why.uncovered == [] and [i.item_id for i in why.ir.items() if not i.item_id.startswith("anc:")] == [
        f"{iep_item.item_id}#inv:1"
    ]
    assert why.ir.audit[-1].explain["reason"] == "invariant:failed_path"
    assert lower(why).all_obligations()[-1]["status"] == "satisfied"

    clock.tick(seconds=60)
    drill = compile_hierarchical(hms.compile(S4_DRILL, filters=scope4, mode="deep"))
    assert [r.req_id for r in drill.requirements] == [
        "identifier:billing_e2e",
        "identifier:subscription_state_regression",
    ]
    assert drill.layers[1].selected == [iep_item.item_id] and drill.layers[1].uncovered == [
        "identifier:subscription_state_regression"
    ]
    assert drill.layers[2].refined is True and len(drill.layers[2].selected) == 1
    l1_item = next(i for i in drill.ir.items() if i.item_id == drill.layers[2].selected[0])
    assert layer_of(l1_item) == Layer.L1_micro and "subscription_state_regression" in l1_item.text
    assert drill.uncovered == [] and lower(drill).ok() is True

    # A requirement nothing in the store can cover is reported, and the lowering says so.
    # 存储中无物可覆盖的需求会被报告，降低结果也如实反映。
    stray = compile_hierarchical(
        deep, requirements=[_req("identifier:nowhere.ts", "identifier", "nowhere.ts", ("nowhere.ts",))]
    )
    assert [r.req_id for r in stray.uncovered] == ["identifier:nowhere.ts"] and all(r.refined for r in stray.layers)
    assert lower(stray).ok() is False


# ---------------------------------------------------------------------------
# Wiring into compile(): selection='coverage' / compile(mode='coverage')
# 接入 compile()：selection='coverage' / compile(mode='coverage')
# ---------------------------------------------------------------------------
FRIDAY = "Never deploy on Fridays."
FRIDAY_QUERY = "why must we never deploy on Fridays"
FRIDAY_REQS = ["constraint:we never deploy on fridays", "entity:fridays"]
COVERAGE_LAYERS = ["L3_macro", "L2_meso", "L1_micro"]
FIVE_OBLIGATIONS = ["evidence_coverage", "no_unresolved_conflict", "governance", "not_revoked", REQUIREMENTS_COVERED]


def test_compile_selection_coverage_replaces_shares_and_lowers_with_requirements_covered(hms, clock, events, scope4):
    """compile(mode='coverage') packs by requirement cover (stats + Coverage section, uncovered empty on S4 + one constraint added after the EPs so no κ repeats it), the default path is byte-identical, and removing the only covering atom makes the requirement uncovered with the PC-IR obligation requirements_covered violated.
    compile(mode='coverage') 按需求覆盖打包（统计 + Coverage 段，S4 + 一条在 EP 之后添加、因而不被任何 κ 复述的约束下 uncovered 为空），默认路径逐字节不变；去掉唯一覆盖它的原子后需求变为未覆盖，PC-IR 义务 requirements_covered 为 violated。

    Input / 输入：
        hms: The HMS fixture (rule LLM + hash embedder + parametrized index).
        hms：HMS 夹具（规则 LLM + 哈希嵌入器 + 参数化索引）。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

    Output / 输出：
        Assertions over retrieval['coverage'] / ['hierarchy'], the card text, to_pcir() and the shares path.
        对 retrieval['coverage'] / ['hierarchy']、卡片文本、to_pcir() 与份额路径的断言。

    State effect / 状态影响：
        None outside the in-memory store (one delete archives the Friday constraint).
        除内存 store 外无副作用（一次 delete 归档 Friday 约束）。
    """
    fixture = events("structured_billing")
    clock.tick(seconds=60)
    hms.add(fixture.constraint["content"], timestamp=fixture.constraint["timestamp"], **scope4)
    clock.tick(seconds=60)
    hms.add_events(fixture.events_fail, **scope4)
    clock.tick(seconds=60)
    hms.add_events(fixture.events_fix, **scope4)
    clock.tick(seconds=60)
    # Added after both EPs closed, so the constraint lives in exactly one atom (no EP κ repeats it).
    # 在两个 EP 闭合之后添加，因此该约束只存在于一个原子中（没有 EP 的 κ 复述它）。
    friday = hms.add(FRIDAY, timestamp="2026-06-15T11:00:00Z", **scope4)["results"][0]
    clock.tick(seconds=60)

    # The default packer is untouched: no coverage keys, no Coverage section, same card twice.
    # 默认打包器不受影响：没有覆盖键、没有 Coverage 段、两次卡片相同。
    shares = hms.compile(FRIDAY_QUERY, filters=scope4, as_of=AS_OF)
    assert hms.compile(FRIDAY_QUERY, filters=scope4, as_of=AS_OF).to_text() == shares.to_text()
    assert "## Coverage" not in shares.to_text() and not {"selection", "coverage", "hierarchy"} & set(shares.retrieval)
    assert [o["name"] for o in (ob.model_dump() for ob in shares.to_pcir().proof_obligations)] == FIVE_OBLIGATIONS[:4]

    cov = hms.compile(FRIDAY_QUERY, filters=scope4, as_of=AS_OF, mode="coverage")
    stats, hierarchy = cov.retrieval["coverage"], cov.retrieval["hierarchy"]
    assert cov.retrieval["selection"] == "coverage" and str(cov.mode) == str(shares.mode)
    assert [r["req_id"] for r in stats["requirements"]] == FRIDAY_REQS
    assert stats["uncovered"] == [] and stats["coverage_ratio"] == 1.0 and set(stats["covered"]) == set(FRIDAY_REQS)
    assert [layer["layer"] for layer in stats["layers"]] == COVERAGE_LAYERS
    assert [layer["layer"] for layer in hierarchy["layers"]] == COVERAGE_LAYERS
    assert stats["covered"][FRIDAY_REQS[0]] == [friday["atom_id"]] and hierarchy["uncovered"] == []
    # Only L1 atoms exist before consolidation: L3 / L2 offered nothing, L1 covered both requirements.
    # 固化之前只有 L1 原子：L3 / L2 无候选，L1 覆盖两条需求。
    assert hierarchy["layers"][2]["selected"] == [friday["atom_id"]] and hierarchy["layers"][2]["refined"] is True
    card = cov.to_text()
    assert card.endswith("## Coverage\n- requirements: 2 | covered: 2 | uncovered: 0")
    assert card.split("\n## Coverage")[0].startswith("# Work State Card\n") and FRIDAY in card
    assert cov.used_tokens <= cov.budget_tokens + cov.retrieval["over_budget_evidence"]
    pcir = cov.to_pcir()
    assert isinstance(pcir, HierarchicalPCIR) and pcir.ok() is True
    assert [o["name"] for o in pcir.all_obligations()] == FIVE_OBLIGATIONS
    assert pcir.all_obligations()[-1]["status"] == "satisfied" and pcir.payload["requirements"]["uncovered"] == []

    # The same selection through the config knob and through the compiler argument; an unknown one is refused.
    # 同一选择方式经配置开关与编译器参数各走一遍；未知的选择方式被拒绝。
    by_arg = hms.compiler.compile(FRIDAY_QUERY, scope=Scope(**scope4), filters=scope4, as_of=AS_OF, selection="coverage")
    assert by_arg.to_text() == card and by_arg.retrieval["coverage"] == stats
    config = hms.config.model_copy(update={"compiler": hms.config.compiler.model_copy(update={"selection": "coverage"})})
    from hms_elite.compiler.compile import StateCompiler

    by_config = StateCompiler(hms.store, hms.index, hms.embedder, hms.graph, config)
    assert by_config.compile(FRIDAY_QUERY, scope=Scope(**scope4), filters=scope4, as_of=AS_OF).to_text() == card
    with pytest.raises(ValidationError) as bad:
        hms.compiler.compile(FRIDAY_QUERY, scope=Scope(**scope4), filters=scope4, as_of=AS_OF, selection="bogus")
    assert bad.value.error_code == "VALIDATION_013"

    # Remove the only covering atom: the requirement stays uncovered and the lowering says so (and raises by default).
    # 去掉唯一覆盖它的原子：需求保持未覆盖，降低结果如实反映（默认情况下抛出）。
    clock.tick(seconds=60)
    hms.delete(friday["id"])
    clock.tick(seconds=60)
    gone = hms.compile(FRIDAY_QUERY, filters=scope4, mode="coverage")
    assert gone.retrieval["coverage"]["uncovered"] == FRIDAY_REQS and gone.retrieval["coverage"]["coverage_ratio"] == 0.0
    assert gone.to_text().endswith(
        "## Coverage\n- requirements: 2 | covered: 0 | uncovered: 2\n"
        f"- uncovered: {FRIDAY_REQS[0]}\n- uncovered: {FRIDAY_REQS[1]}"
    )
    failed = gone.to_pcir(require_witness=False)
    assert failed.ok() is False and failed.all_obligations()[-1]["status"] == "violated"
    assert [r["req_id"] for r in failed.all_obligations()[-1]["details"]["uncovered"]] == FRIDAY_REQS
    assert failed.all_obligations()[0]["status"] == "satisfied"
    with pytest.raises(ProofObligationError) as raised:
        gone.to_pcir()
    assert raised.value.details["violated"] == [REQUIREMENTS_COVERED]
    assert raised.value.details["uncovered_requirements"] == FRIDAY_REQS


def test_compile_selection_coverage_refines_interface_first_after_consolidation(hms, clock, events, scope4):
    """After consolidate, compile(mode='coverage') on the S4 goal covers with the L2 InterfaceEP and never drills L1.
    固化之后，对 S4 目标的 compile(mode='coverage') 用 L2 InterfaceEP 覆盖，绝不下钻 L1。

    Input / 输入：
        hms: The HMS fixture.
        hms：HMS 夹具。

        clock: FrozenClock fixture.
        clock：FrozenClock 夹具。

        events: Fixture loader.
        events：夹具加载器。

        scope4: The S4 scope dict.
        scope4：S4 范围字典。

    Output / 输出：
        Assertions over the layer reports of the wired compile against compile_hierarchical on the deep card.
        对接线后编译的层报告与 deep 卡片上 compile_hierarchical 结果的断言。

    State effect / 状态影响：
        None outside the in-memory store.
        除内存 store 外无副作用。
    """
    _ingest_s4(hms, clock, events, scope4)
    clock.tick(seconds=60)
    wired = hms.compile(S4_QUERY, filters=scope4, mode="coverage", as_of=AS_OF)
    manual = compile_hierarchical(hms.compile(S4_QUERY, filters=scope4, mode="deep", as_of=AS_OF))
    layers = wired.retrieval["hierarchy"]["layers"]
    assert [layer["layer"] for layer in layers] == COVERAGE_LAYERS
    assert layers[1]["selected"] == manual.layers[1].selected and len(layers[1]["selected"]) == 1
    assert layers[2]["selected"] == [] and layers[2]["refined"] is False
    assert wired.retrieval["coverage"]["uncovered"] == [] and wired.retrieval["coverage"]["kappa_closure"]
    iep_item = next(i for i in wired.episodes if str(i.scale_level) == "L2_meso")
    assert layers[1]["selected"] == [iep_item.item_id]
    assert wired.to_pcir().ok() is True and "## Coverage\n- requirements: 1 | covered: 1 | uncovered: 0" in wired.to_text()
