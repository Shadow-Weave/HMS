"""The nine HMS §10 benchmark tasks as executable scenario + query + explained score, plus the LongMemEval adapter.
HMS §10 的九个基准任务（可执行的场景 + 查询 + 可解释打分），以及 LongMemEval 适配器骨架。
"""

# Purpose (EN): Give every row of the HMS §10 task table (preference recall, temporal, multi-session, handoff,
#   rollback, conflict / revision, evidence audit, skill distillation, world-model transition) a Task object that
#   builds its scenario from tests/fixtures/events, queries through a Probe and scores 0–1 with reasons.
# 目的（中）：为 HMS §10 任务表的每一行（偏好回忆、时间、多会话、交接、回滚、冲突 / 修订、证据审计、技能蒸馏、
#   世界模型转移）提供一个 Task 对象：由 tests/fixtures/events 构建场景，经 Probe 查询，给出 0–1 分与原因。

# Input (EN): tests/fixtures/events/*.json (chat_preference, chat_multisession, chat_conflict, structured_billing),
#   an HMS to build on and a Probe to query through; a LongMemEval JSON file for the adapter.
# 输入（中）：tests/fixtures/events/*.json（chat_preference、chat_multisession、chat_conflict、structured_billing）、
#   用于构建的 HMS 与用于查询的 Probe；适配器另需一个 LongMemEval JSON 文件。

# Output (EN): TASKS (nine Task objects in §10 order), longmemeval_tasks(path) and the fixture constants.
# 输出（中）：TASKS（按 §10 顺序的九个 Task）、longmemeval_tasks(path) 以及夹具常量。

# State effect (EN): Task.build writes the scenario into the HMS it is given (add / add_events / checkpoint / feedback /
#   rollback / revoke / consolidate); Task.run only queries through the Probe.
# 状态影响（中）：Task.build 把场景写入给定的 HMS（add / add_events / checkpoint / feedback / rollback / revoke /
#   consolidate）；Task.run 只经 Probe 查询。

from __future__ import annotations

import json
import os
import pathlib
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from hms_elite.enums import EdgeType, ObligationName, ObligationStatus, Reward, TransitionKind
from hms_elite.eval.ablation import Check, Probe, Task, TaskScore
from hms_elite.models import ContextIR

# Fixture directory shared with the test suite (tests/fixtures/events).
# 与测试套件共享的夹具目录（tests/fixtures/events）。
FIXTURES_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "events"

# Scenario constants (CONTRACT §19 S1–S6 texts; the rule LLM produces exactly these).
# 场景常量（契约 §19 S1–S6 的文本；规则 LLM 恰好产出这些）。
USER = "alice"
S1_SCOPE = {"user_id": USER, "run_id": "run-coffee"}
LATTE = "User likes oat-milk lattes in the morning."
BLACK = "User likes black coffee."
SARAH_1 = "Sarah joined the billing team in March."
SARAH_2 = "Sarah shipped the webhook retry design on May 6."
SARAH_3 = "Sarah is reviewing the checkout refactor this week."
S4_QUERY = "fix checkout webhook retry"
UC_TEXT = "Do not modify the subscription state machine."
EP1_A = "patch src/billing/retry.ts: add exponential backoff"
EP1_Y = "billing_e2e failed: subscription_state_regression"
EP2_A = "patch src/billing/idempotency.ts: add idempotency key"
EP2_Y = "billing_e2e passed"
DC_TEXT = f"Do not re-apply '{EP1_A}': {EP1_Y}"
CORRECTION = "retry.ts may change once billing_e2e is isolated."
SKILL_TEXT = f"Skill: {UC_TEXT}"
FAILED_PATH_INVARIANT = f"do not: {EP1_A}"
KAPPA_KEYS = frozenset({"constraints", "visibility", "permission_scope", "provenance"})
PARIS = "User live in Paris."
BERLIN = "User live in Berlin."

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


# ---------------------------------------------------------------------------
# Fixture and scenario helpers
# 夹具与场景辅助
# ---------------------------------------------------------------------------
def load_fixture(name: str) -> Any:
    """Load tests/fixtures/events/<name>.json (a bare list is returned as-is).
    加载 tests/fixtures/events/<name>.json（裸列表按原样返回）。

    Input / 输入：
        name: Fixture stem without .json.
        name：不含 .json 的夹具名。

    Output / 输出：
        The parsed JSON (list or dict).
        解析后的 JSON（列表或字典）。

    State effect / 状态影响：
        None.
        无。
    """
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _add_messages(hms: Any, messages: Iterable[dict], **scope: Any) -> list[dict]:
    """Add fixture messages one by one (timestamp / run_id taken from the message when present).
    逐条加入夹具消息（消息自带 timestamp / run_id 时优先使用）。
    """
    rows: list[dict] = []
    for msg in messages:
        keys = dict(scope)
        if msg.get("run_id"):
            keys["run_id"] = msg["run_id"]
        rows.extend(hms.add(msg["content"], timestamp=msg.get("timestamp"), **keys)["results"])
    return rows


def _build_s4(hms: Any, *, fix: bool = True) -> dict:
    """Ingest the S4 billing scenario (constraint + events_fail [+ events_fix]); return the ids a task needs.
    摄取 S4 计费场景（约束 + events_fail [+ events_fix]）；返回任务所需的 id。
    """
    fixture = load_fixture("structured_billing")
    scope = dict(fixture["scope"])
    r0 = hms.add(fixture["constraint"]["content"], timestamp=fixture["constraint"]["timestamp"], **scope)
    ra = hms.add_events(fixture["events_fail"], **scope)
    ctx: dict[str, Any] = {
        "filters": scope,
        "uc": r0["results"][0]["atom_id"],
        "mem_uc": r0["results"][0]["id"],
        "anc_uc": r0["results"][0]["anchor_ids"][0],
        "anc_e1": ra["events"][0]["anchor_id"],
        "anc_e2": ra["events"][1]["anchor_id"],
        "ep1": ra["eps"][0],
        "dc": ra["constraints"][0],
    }
    if fix:
        rb = hms.add_events(fixture["events_fix"], **scope)
        ctx.update(anc_e3=rb["events"][0]["anchor_id"], anc_e4=rb["events"][1]["anchor_id"], ep2=rb["eps"][0])
    return ctx


def _checkpoint_feedback_rollback(hms: Any, ctx: dict) -> dict:
    """S6 on top of S4: checkpoint → state_outdated feedback on the user constraint → rollback.
    在 S4 之上的 S6：检查点 → 对用户约束的 state_outdated 反馈 → 回滚。
    """
    cp = hms.checkpoint(filters=ctx["filters"], label="before-feedback")
    fb = hms.feedback(ctx["mem_uc"], "state_outdated", correction=CORRECTION)
    rb = hms.rollback(cp)
    ctx.update(checkpoint=cp, uc2=fb["actions"][1]["atom_id"], rollback=rb)
    return ctx


def _texts(items: Iterable[Any]) -> list[str]:
    """Item / row texts in order.
    按顺序的条目 / 行文本。
    """
    out: list[str] = []
    for item in items:
        out.append(item["memory"] if isinstance(item, dict) else item.text)
    return out


def _check(name: str, hit: bool, detail: str) -> Check:
    """A binary Check.
    二值检查项。
    """
    return Check(name=name, score=1.0 if hit else 0.0, detail=detail)


def _off(name: str, what: str) -> Check:
    """A miss because the ablation switched the needed plane off.
    因消融关闭了所需平面而未命中。
    """
    return Check(name=name, score=0.0, detail=f"{what} unavailable in this ablation")


def _evidence(ir: ContextIR) -> list[Any]:
    """Evidence audit items of a ContextIR.
    ContextIR 的证据审计项。
    """
    return [i for i in ir.audit if i.explain.get("kind") == "evidence"]


def _revised(ir: ContextIR) -> list[Any]:
    """Audit items whose reason is 'revised'.
    原因为 'revised' 的审计项。
    """
    return [i for i in ir.audit if i.explain.get("reason") == "revised"]


def _obligation_check(name: str, probe: Probe, ir: ContextIR, *, expect_ok: bool, obligation: str | None) -> Check:
    """Check PC-IR obligations: all four present and ok() == expect_ok (or one named obligation's status).
    检查 PC-IR 义务：四条齐全且 ok() == expect_ok（或某条命名义务的状态）。
    """
    pcir = probe.pcir(ir)
    if pcir is None:
        return _off(name, "PC-IR proof obligations")
    statuses = {str(o.name): str(o.status) for o in pcir.proof_obligations}
    complete = set(statuses) == {o.value for o in ObligationName}
    if obligation is not None:
        status = statuses.get(obligation)
        wanted = ObligationStatus.satisfied.value if expect_ok else ObligationStatus.violated.value
        return _check(name, complete and status == wanted, f"{obligation}={status} (wanted {wanted})")
    return _check(name, complete and pcir.ok() == expect_ok, f"obligations={statuses} ok={pcir.ok()}")


# ---------------------------------------------------------------------------
# 1. preference recall
# 1. 偏好回忆
# ---------------------------------------------------------------------------
def build_preference_recall(hms: Any) -> dict:
    """S1: two coffee preferences, the second revising the first at ingest time.
    S1：两条咖啡偏好，第二条在摄取时修订第一条。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        Context {'filters'}.
        上下文 {'filters'}。

    State effect / 状态影响：
        Adds the chat_preference fixture.
        加入 chat_preference 夹具。
    """
    _add_messages(hms, load_fixture("chat_preference"), **S1_SCOPE)
    return {"filters": dict(S1_SCOPE)}


def run_preference_recall(probe: Probe, ctx: dict) -> TaskScore:
    """Ask for the coffee preference: newest wins, the revised one is gone, the axis projection says 'black coffee'.
    询问咖啡偏好：最新者胜出、被修订者消失、轴投影给出 'black coffee'。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: Build context.
        ctx：构建上下文。

    Output / 输出：
        TaskScore over four checks.
        四项检查的 TaskScore。

    State effect / 状态影响：
        None.
        无。
    """
    query = "what coffee does the user like"
    hits = _texts(probe.search(query, filters={"user_id": USER}))
    ir = probe.compile(query, filters=ctx["filters"])
    latest = probe.axis_query(
        {
            "filters": {"user_id": USER, "topic": "coffee"},
            "project_axis": "value",
            "scale_constraint": ["state"],
            "aggregation": "latest",
        }
    )
    checks = [
        _check("top_hit_is_newest", bool(hits) and hits[0] == BLACK, f"top hit: {hits[:1]}"),
        _check("revised_not_live", bool(hits) and LATTE not in hits, f"hits: {hits}"),
        _check("preference_slot", _texts(ir.preferences) == [BLACK], f"preferences: {_texts(ir.preferences)}"),
        _off("axis_latest_projection", "axis projection")
        if latest is None
        else _check(
            "axis_latest_projection", latest.projection == [["black coffee"]], f"projection: {latest.projection}"
        ),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 2. temporal
# 2. 时间
# ---------------------------------------------------------------------------
def build_multisession(hms: Any) -> dict:
    """S2/S3: three Sarah messages under run-a / run-b / run-c with explicit timestamps.
    S2/S3：三条 Sarah 消息，分属 run-a / run-b / run-c，带显式时间戳。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        Context {'filters'}.
        上下文 {'filters'}。

    State effect / 状态影响：
        Adds the chat_multisession fixture.
        加入 chat_multisession 夹具。
    """
    _add_messages(hms, load_fixture("chat_multisession"), user_id=USER)
    return {"filters": {"user_id": USER}}


def run_temporal(probe: Probe, ctx: dict) -> TaskScore:
    """'what happened in May' must return exactly the May event; the entity timeline must be time-ordered.
    'what happened in May' 必须恰好返回五月事件；实体时间线必须按时间排序。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: Build context.
        ctx：构建上下文。

    Output / 输出：
        TaskScore over four checks.
        四项检查的 TaskScore。

    State effect / 状态影响：
        None.
        无。
    """
    query = "what happened in May"
    hits = _texts(probe.search(query, filters=ctx["filters"]))
    ir = probe.compile(query, filters=ctx["filters"])
    timeline = _texts(probe.recall(filters={**ctx["filters"], "entity": "Sarah"}))
    projection = probe.axis_query(
        {"filters": {**ctx["filters"], "entity": "Sarah"}, "project_axis": "time", "aggregation": "list"}
    )
    ascending = None
    if projection is not None and isinstance(projection.projection, list):
        times = [pair[0] for pair in projection.projection]
        ascending = times == sorted(times) and len(times) == 3
    checks = [
        _check("time_gated_search", hits == [SARAH_2], f"hits: {hits}"),
        _check("time_gated_compile", _texts(ir.facts) == [SARAH_2], f"facts: {_texts(ir.facts)}"),
        _check("timeline_newest_first", timeline == [SARAH_3, SARAH_2, SARAH_1], f"timeline: {timeline}"),
        _off("time_axis_projection", "time axis projection")
        if ascending is None
        else _check("time_axis_projection", ascending, f"projection: {projection.projection}"),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 3. multi-session
# 3. 多会话
# ---------------------------------------------------------------------------
def run_multi_session(probe: Probe, ctx: dict) -> TaskScore:
    """Everything about Sarah across three runs, one run isolated, every hit carrying the entity key.
    跨三个 run 的全部 Sarah 信息、单个 run 的隔离、每个命中都带实体键。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: Build context.
        ctx：构建上下文。

    Output / 输出：
        TaskScore over four checks.
        四项检查的 TaskScore。

    State effect / 状态影响：
        None.
        无。
    """
    expected = {SARAH_1, SARAH_2, SARAH_3}
    rows = probe.search("Sarah", filters=ctx["filters"])
    hits = _texts(rows)
    run_b = _texts(probe.recall(filters={**ctx["filters"], "run_id": "run-b", "entity": "sarah"}))
    ir = probe.compile("what do we know about Sarah", filters=ctx["filters"])
    keyed = bool(rows) and all("sarah" in r.get("hms", {}).get("entity_ids", []) for r in rows)
    checks = [
        _check("all_sessions_recalled", set(hits) == expected, f"hits: {hits}"),
        _check("run_isolation", run_b == [SARAH_2], f"run-b: {run_b}"),
        _check("compile_aggregates_entity", set(_texts(ir.facts)) == expected, f"facts: {_texts(ir.facts)}"),
        _check("entity_key_on_hits", keyed, f"entity ids: {[r.get('hms', {}).get('entity_ids') for r in rows]}"),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 4. handoff
# 4. 交接
# ---------------------------------------------------------------------------
def build_handoff(hms: Any) -> dict:
    """S4: user constraint, failed retry.ts attempt, successful idempotency.ts attempt.
    S4：用户约束、失败的 retry.ts 尝试、成功的 idempotency.ts 尝试。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        S4 context (filters, atom / anchor / EP ids).
        S4 上下文（filters、原子 / 锚点 / EP id）。

    State effect / 状态影响：
        Ingests the structured_billing fixture.
        摄取 structured_billing 夹具。
    """
    return _build_s4(hms, fix=True)


def _episode_texts(ir: ContextIR, reward: str) -> list[str]:
    """Episode item actions carrying a reward.
    带某回报的情节项动作。
    """
    return [str(i.explain.get("a")) for i in ir.episodes if str(i.explain.get("r")) == reward]


def run_handoff(probe: Probe, ctx: dict) -> TaskScore:
    """The receiving agent needs constraints, the failed path, the working path, a next action and its evidence.
    接手方需要约束、失败路径、有效路径、下一步动作及其证据。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: S4 context.
        ctx：S4 上下文。

    Output / 输出：
        TaskScore over seven checks.
        七项检查的 TaskScore。

    State effect / 状态影响：
        None.
        无。
    """
    ir = probe.compile(S4_QUERY, filters=ctx["filters"], budget_tokens=800)
    constraints = _texts(ir.constraints)
    refs = {str(i.explain.get("source_ref")) for i in _evidence(ir)}
    checks = [
        _check("user_constraint", UC_TEXT in constraints, f"constraints: {constraints}"),
        _check("derived_constraint", DC_TEXT in constraints, f"constraints: {constraints}"),
        _check(
            "failed_path",
            EP1_A in _episode_texts(ir, Reward.negative.value),
            f"negative: {_episode_texts(ir, 'negative')}",
        ),
        _check(
            "working_path",
            EP2_A in _episode_texts(ir, Reward.positive.value),
            f"positive: {_episode_texts(ir, 'positive')}",
        ),
        _check("next_action", str(ir.next_action).startswith(f"Continue from '{EP2_A}'"), f"next: {ir.next_action}"),
        _check(
            "evidence_refs",
            {"file://src/billing/retry.ts", "test://billing_e2e", "file://src/billing/idempotency.ts"} <= refs,
            f"refs: {sorted(refs)}",
        ),
        _obligation_check("proof_obligations", probe, ir, expect_ok=True, obligation=None),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 5. rollback
# 5. 回滚
# ---------------------------------------------------------------------------
def build_rollback(hms: Any) -> dict:
    """S4 + S6: checkpoint, feedback revising the constraint, rollback to the checkpoint.
    S4 + S6：检查点、修订约束的反馈、回滚到检查点。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        Context with checkpoint, uc2 and the rollback report.
        含检查点、uc2 与回滚报告的上下文。

    State effect / 状态影响：
        Ingests S4 then writes checkpoint, feedback and rollback rows.
        摄取 S4 后写入检查点、反馈与回滚行。
    """
    return _checkpoint_feedback_rollback(hms, _build_s4(hms, fix=True))


def run_rollback(probe: Probe, ctx: dict) -> TaskScore:
    """After rollback the original constraint is live, the correction is archived-and-explained, the edge is visible.
    回滚后原约束存活、修正被归档且可解释、回滚边可见。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: Rollback context.
        ctx：回滚上下文。

    Output / 输出：
        TaskScore over five checks.
        五项检查的 TaskScore。

    State effect / 状态影响：
        None (transition export is idempotent).
        无（转移导出幂等）。
    """
    ir = probe.compile(S4_QUERY, filters=ctx["filters"])
    constraints = _texts(ir.constraints)
    revised = [(i.text, i.explain.get("superseded_by_text")) for i in _revised(ir)]
    trace = probe.why(ctx["uc"], hops=2)
    labels = [] if trace is None else [str(e.get("label")) for e in trace["edges"]]
    rollback_edge = any(label.startswith("rollback:") for label in labels)
    transitions = probe.transitions(filters=ctx["filters"])
    rollbacks = [t for t in transitions if t["source_kind"] == TransitionKind.rollback.value]
    checks = [
        _check("restored_constraint_live", UC_TEXT in constraints, f"constraints: {constraints}"),
        _check("correction_not_live", CORRECTION not in constraints, f"constraints: {constraints}"),
        _check("correction_explained_as_revised", (CORRECTION, UC_TEXT) in revised, f"revised: {revised}"),
        _off("rollback_edge_visible", "graph traversal")
        if trace is None
        else _check("rollback_edge_visible", rollback_edge, f"edge labels: {labels}"),
        _check(
            "rollback_transition",
            any(t["a_t"] == "rollback" and t["s_t1"].get("text") == UC_TEXT for t in rollbacks),
            f"rollback transitions: {len(rollbacks)} of kinds {sorted({t['source_kind'] for t in transitions})}",
        ),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 6. conflict / revision
# 6. 冲突 / 修订
# ---------------------------------------------------------------------------
def build_conflict_revision(hms: Any) -> dict:
    """An unresolved residence conflict (Paris vs Berlin, out-of-order timestamps) plus the S1 preference revision.
    一个未解决的居住地冲突（Paris 与 Berlin，时间戳乱序）加上 S1 的偏好修订。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        Context {'filters', 'pref_filters', 'paris', 'berlin'}.
        上下文 {'filters', 'pref_filters', 'paris', 'berlin'}。

    State effect / 状态影响：
        Adds the chat_conflict and chat_preference fixtures.
        加入 chat_conflict 与 chat_preference 夹具。
    """
    conflict_scope = {"user_id": USER, "run_id": "run-home"}
    rows = _add_messages(hms, load_fixture("chat_conflict"), **conflict_scope)
    _add_messages(hms, load_fixture("chat_preference"), **S1_SCOPE)
    paris = next(r["atom_id"] for r in rows if r["memory"] == PARIS)
    berlin = next(r["atom_id"] for r in rows if r["memory"] == BERLIN)
    return {"filters": conflict_scope, "pref_filters": dict(S1_SCOPE), "paris": paris, "berlin": berlin}


def run_conflict_revision(probe: Probe, ctx: dict) -> TaskScore:
    """The conflict must be surfaced (not silently answered) and the revision must be explained.
    冲突必须被揭示（而非静默作答），修订必须可解释。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: Conflict context.
        ctx：冲突上下文。

    Output / 输出：
        TaskScore over six checks.
        六项检查的 TaskScore。

    State effect / 状态影响：
        None.
        无。
    """
    ir = probe.compile("where does the user live", filters=ctx["filters"])
    # The conflict edge points older → newer-ingested (Paris → Berlin); why() walks incoming edges, so ask at Berlin.
    # 冲突边由先摄取指向后摄取（Paris → Berlin）；why() 沿入边回溯，因此在 Berlin 处提问。
    trace = probe.why(ctx["berlin"], hops=1)
    types = [] if trace is None else [str(e["edge_type"]) for e in trace["edges"]]
    pref = probe.compile("coffee", filters=ctx["pref_filters"])
    revised = [(i.text, i.explain.get("superseded_by_text")) for i in _revised(pref)]
    hits = _texts(probe.search("coffee", filters=ctx["pref_filters"]))
    checks = [
        _check("conflict_surfaced", len(ir.unresolved_conflicts) == 1, f"unresolved: {ir.unresolved_conflicts}"),
        _obligation_check(
            "conflict_obligation_violated", probe, ir, expect_ok=False, obligation="no_unresolved_conflict"
        ),
        _off("conflict_edge_visible", "graph traversal")
        if trace is None
        else _check("conflict_edge_visible", EdgeType.conflict.value in types, f"edge types: {types}"),
        _check("revision_newest_live", _texts(pref.preferences) == [BLACK], f"preferences: {_texts(pref.preferences)}"),
        _check("revision_explained", (LATTE, BLACK) in revised, f"revised: {revised}"),
        _check("revised_not_in_recall", bool(hits) and LATTE not in hits, f"hits: {hits}"),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 7. evidence audit
# 7. 证据审计
# ---------------------------------------------------------------------------
def build_evidence_audit(hms: Any) -> dict:
    """S4 then revoke the idempotency.ts diff anchor: one witness of the working path becomes disputed.
    S4 后撤销 idempotency.ts diff 的锚点：有效路径的一个见证变为争议。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        S4 context plus 'revoked' (the revoked anchor id).
        S4 上下文加 'revoked'（被撤销的锚点 id）。

    State effect / 状态影响：
        Ingests S4 and revokes one anchor (cascade archives the event atom).
        摄取 S4 并撤销一个锚点（级联归档事件原子）。
    """
    ctx = _build_s4(hms, fix=True)
    hms.revoke(ctx["anc_e3"], reason="evidence audit")
    ctx["revoked"] = ctx["anc_e3"]
    return ctx


def run_evidence_audit(probe: Probe, ctx: dict) -> TaskScore:
    """Every claim keeps an undisputed witness, the revoked anchor is marked, edges carry witnesses, obligations hold.
    每条陈述保有无争议见证、被撤销锚点被标记、边带见证、义务成立。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: Evidence context.
        ctx：证据上下文。

    Output / 输出：
        TaskScore over six checks.
        六项检查的 TaskScore。

    State effect / 状态影响：
        None.
        无。
    """
    ir = probe.compile(S4_QUERY, filters=ctx["filters"])
    claims = [i for i in ir.items() if i.slot != "audit"]
    undisputed = [[a for a in i.anchor_ids if a not in i.explain.get("disputed_anchor_ids", [])] for i in claims]
    witnessed = bool(claims) and all(undisputed)
    dc = [i for i in ir.constraints if i.text == DC_TEXT]
    dc_anchors = set(dc[0].anchor_ids) if dc else set()
    marked = [i for i in _evidence(ir) if i.anchor_ids == [ctx["revoked"]]]
    revoked_marked = bool(marked) and marked[0].explain.get("revoked_at") is not None
    ep2 = [i for i in ir.episodes if i.explain.get("a") == EP2_A]
    disputed = ep2[0].explain.get("disputed_anchor_ids", []) if ep2 else []
    trace = probe.why(ctx["dc"], hops=2)
    edges_witnessed = (
        trace is not None and bool(trace["edges"]) and all(e["witness_anchor_ids"] for e in trace["edges"])
    )
    checks = [
        _check(
            "every_claim_has_undisputed_witness", witnessed, f"claims: {len(claims)}, empty: {undisputed.count([])}"
        ),
        _check(
            "derived_constraint_witnesses",
            dc_anchors == {ctx["anc_e2"], ctx["anc_e1"]},
            f"dc anchors: {sorted(dc_anchors)}",
        ),
        _check("revoked_anchor_marked", revoked_marked, f"evidence rows for revoked anchor: {len(marked)}"),
        _check("dispute_propagated_to_claim", ctx["revoked"] in disputed, f"ep2 disputed: {disputed}"),
        _off("edges_carry_witnesses", "graph traversal")
        if trace is None
        else _check("edges_carry_witnesses", edges_witnessed, f"edges: {len(trace['edges'])}"),
        _obligation_check("proof_obligations", probe, ir, expect_ok=True, obligation=None),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 8. skill distillation
# 8. 技能蒸馏
# ---------------------------------------------------------------------------
def build_skill_distillation(hms: Any) -> dict:
    """S4 then consolidate(): coarsen to an InterfaceEP, project its edges, promote the constraint to a skill.
    S4 后 consolidate()：粗化为 InterfaceEP、投影其边、把约束提升为技能。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        S4 context plus 'skill' (the skill atom id, or None when none was promoted).
        S4 上下文加 'skill'（技能原子 id；未提升时为 None）。

    State effect / 状态影响：
        Ingests S4 and runs the scheduler phases on its scope.
        摄取 S4 并在其范围上运行调度器阶段。
    """
    ctx = _build_s4(hms, fix=True)
    hms.consolidate(filters=ctx["filters"])
    rows = hms.get_all(filters={**ctx["filters"], "scale": "L3_macro"})["results"]
    skills = [r for r in rows if r["hms"].get("source_kind") == "skill"]
    ctx["skill"] = skills[0]["atom_id"] if skills else None
    return ctx


def run_skill_distillation(probe: Probe, ctx: dict) -> TaskScore:
    """The skill is reusable, traceable to its EPs through the InterfaceEP, and its failed path is inherited.
    技能可复用、可经 InterfaceEP 追溯到其 EP、其失败路径可继承。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: Skill context.
        ctx：技能上下文。

    Output / 输出：
        TaskScore over four checks.
        四项检查的 TaskScore。

    State effect / 状态影响：
        None.
        无。
    """
    ir = probe.compile(S4_QUERY, filters=ctx["filters"])
    skills = _texts(ir.skills)
    trace = None if ctx["skill"] is None else probe.why(ctx["skill"], hops=2)
    distill = trace is not None and any(e["edge_type"] == EdgeType.distill.value for e in trace["edges"])
    ep_refs = set() if trace is None else {n["ref_id"] for n in trace["nodes"] if n["node_kind"] == "ep"}
    support = {ctx["ep1"], ctx["ep2"]} <= ep_refs
    reasons = [str(i.explain.get("reason")) for i in ir.audit]
    checks = [
        _check("skill_reusable", skills == [SKILL_TEXT], f"skills: {skills}"),
        _off("distill_edge_visible", "graph traversal")
        if trace is None
        else _check("distill_edge_visible", distill, f"edge types: {[e['edge_type'] for e in trace['edges']]}"),
        _off("support_eps_reachable", "graph traversal")
        if trace is None
        else _check("support_eps_reachable", support, f"ep nodes: {sorted(ep_refs)}"),
        _check("failed_path_invariant_inherited", "invariant:failed_path" in reasons, f"audit reasons: {reasons}"),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# 9. world-model transition
# 9. 世界模型转移
# ---------------------------------------------------------------------------
def build_world_model(hms: Any) -> dict:
    """S4 + S6 + consolidate(): EP, causal, revision, rollback and InterfaceEP transitions all exist.
    S4 + S6 + consolidate()：EP、causal、revision、rollback 与 InterfaceEP 转移全部存在。

    Input / 输入：
        hms: A fresh HMS.
        hms：新的 HMS。

    Output / 输出：
        Rollback context.
        回滚上下文。

    State effect / 状态影响：
        Ingests S4, runs S6 and the scheduler.
        摄取 S4、运行 S6 与调度器。
    """
    ctx = _checkpoint_feedback_rollback(hms, _build_s4(hms, fix=True))
    hms.consolidate(filters=ctx["filters"])
    return ctx


def run_world_model(probe: Probe, ctx: dict) -> TaskScore:
    """Export (S_t, a_t, S_t+1, r_t, κ_t, w_t) and check every transition kind the scenario produced.
    导出 (S_t, a_t, S_t+1, r_t, κ_t, w_t) 并检查场景产出的每种转移。

    Input / 输入：
        probe: The ablation probe.
        probe：消融探针。

        ctx: World-model context.
        ctx：世界模型上下文。

    Output / 输出：
        TaskScore over six checks.
        六项检查的 TaskScore。

    State effect / 状态影响：
        None (transition export is idempotent).
        无（转移导出幂等）。
    """
    rows = probe.transitions(filters=ctx["filters"])
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(str(row["source_kind"]), []).append(row)
    eps = {(t["a_t"], t["r_t"]) for t in by_kind.get("ep", [])}
    causal = any(
        t["s_t"].get("ep_id") == ctx["ep1"] and t["s_t1"].get("ep_id") == ctx["ep2"] for t in by_kind.get("causal", [])
    )
    revision = any(t["a_t"] == "revise" and t["s_t1"].get("text") == CORRECTION for t in by_kind.get("revision", []))
    rollback = any(t["a_t"] == "rollback" and t["s_t1"].get("text") == UC_TEXT for t in by_kind.get("rollback", []))
    iep = any(FAILED_PATH_INVARIANT in t["s_t1"].get("invariants", []) for t in by_kind.get("interface_ep", []))
    well_formed = bool(rows) and all(t["w_t"] and set(t["kappa_t"]) == KAPPA_KEYS for t in rows)
    kinds = sorted(by_kind)
    checks = [
        _check("ep_transitions", {(EP1_A, "negative"), (EP2_A, "positive")} <= eps, f"ep (a, r): {sorted(eps)}"),
        _check("causal_transition", causal, f"kinds: {kinds}"),
        _check("revision_transition", revision, f"kinds: {kinds}"),
        _check("rollback_transition", rollback, f"kinds: {kinds}"),
        _check("interface_ep_transition", iep, f"kinds: {kinds}"),
        _check("witness_and_kappa_complete", well_formed, f"rows: {len(rows)}"),
    ]
    return TaskScore.from_checks(checks)


# ---------------------------------------------------------------------------
# The §10 task table
# §10 任务表
# ---------------------------------------------------------------------------
TASKS: tuple[Task, ...] = (
    Task(
        "preference_recall",
        "S1 coffee preference with an ingest-time revision",
        build_preference_recall,
        run_preference_recall,
    ),
    Task("temporal", "S2 'what happened in May' over three dated Sarah events", build_multisession, run_temporal),
    Task("multi_session", "S3 Sarah across run-a / run-b / run-c", build_multisession, run_multi_session),
    Task("handoff", "S4 Work State handed to a new agent", build_handoff, run_handoff),
    Task("rollback", "S6 checkpoint → feedback → rollback", build_rollback, run_rollback),
    Task(
        "conflict_revision",
        "unresolved residence conflict + S1 revision",
        build_conflict_revision,
        run_conflict_revision,
    ),
    Task("evidence_audit", "S4 with one revoked witness", build_evidence_audit, run_evidence_audit),
    Task(
        "skill_distillation",
        "S4 consolidated into an InterfaceEP and a skill",
        build_skill_distillation,
        run_skill_distillation,
    ),
    Task("world_model_transition", "S4 + S6 + consolidate transition export", build_world_model, run_world_model),
)
TASK_BY_NAME: dict[str, Task] = {t.name: t for t in TASKS}


# ---------------------------------------------------------------------------
# LongMemEval adapter skeleton (no download; skipped when the file is absent)
# LongMemEval 适配器骨架（不下载；文件缺失时跳过）
# ---------------------------------------------------------------------------
_LME_DATE_FORMATS = ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def _lme_timestamp(value: Any) -> str | None:
    """Parse a LongMemEval session date ('2023/05/20 (Sat) 02:21' or ISO) into a UTC ISO instant; None when unknown.
    将 LongMemEval 会话日期（'2023/05/20 (Sat) 02:21' 或 ISO）解析为 UTC ISO 时刻；无法解析时 None。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in _LME_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()


def _answer_tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens and CJK characters of an answer.
    答案的小写字母数字词元与中日韩字符。
    """
    return _TOKEN_RE.findall(str(text).lower())


def task_from_longmemeval(entry: dict, *, top_k: int = 5) -> Task:
    """Turn one LongMemEval entry into a Task: sessions are ingested as dated chats, the question is scored by coverage.
    将一条 LongMemEval 条目转成 Task：会话按带日期的聊天摄取，问题按答案覆盖率打分。

    Input / 输入：
        entry: {'question_id', 'question', 'answer', 'haystack_sessions': [[{role, content}]], 'haystack_dates', ...}.
        entry：{'question_id', 'question', 'answer', 'haystack_sessions': [[{role, content}]], 'haystack_dates', …}。

        top_k: Search rows consulted for the answer.
        top_k：用于回答的搜索行数。

    Output / 输出：
        Task named 'longmemeval:<question_id>' (score = answer-token coverage of hits ∪ compiled facts, in [0, 1]).
        名为 'longmemeval:<question_id>' 的 Task（分数 = 命中 ∪ 编译事实对答案词元的覆盖率，[0, 1]）。

    State effect / 状态影响：
        None here; the returned Task's build ingests into the HMS it receives.
        此处无；返回的 Task 的 build 会摄取到收到的 HMS 中。
    """
    qid = str(entry.get("question_id") or "unknown")
    question = str(entry.get("question") or "")
    answer = str(entry.get("answer") or "")
    sessions = list(entry.get("haystack_sessions") or [])
    dates = list(entry.get("haystack_dates") or [])
    scope = {"user_id": f"lme-{qid}"}

    def build(hms: Any) -> dict:
        """Ingest every haystack session as one chat under the question's user scope.
        把每个 haystack 会话作为一次聊天摄取到该问题的用户范围下。
        """
        for index, session in enumerate(sessions):
            messages = [
                {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
                for m in session
                if isinstance(m, dict) and str(m.get("content") or "").strip()
            ]
            if not messages:
                continue
            timestamp = _lme_timestamp(dates[index]) if index < len(dates) else None
            hms.add(messages, timestamp=timestamp, run_id=f"session-{index}", **scope)
        return {"filters": dict(scope), "question": question, "answer": answer}

    def run(probe: Probe, ctx: dict) -> TaskScore:
        """Coverage of the answer tokens by the top-k hits and the compiled facts.
        top-k 命中与编译事实对答案词元的覆盖率。
        """
        hits = _texts(probe.search(ctx["question"], filters=ctx["filters"], top_k=top_k))
        ir = probe.compile(ctx["question"], filters=ctx["filters"])
        corpus = " ".join(hits + _texts(ir.facts) + _texts(ir.preferences) + _texts(ir.episodes)).lower()
        tokens = _answer_tokens(ctx["answer"])
        if not tokens:
            return TaskScore.from_checks([Check("answer_coverage", 0.0, "empty answer")])
        covered = [t for t in tokens if t in corpus]
        score = len(covered) / len(tokens)
        return TaskScore.from_checks([Check("answer_coverage", score, f"{len(covered)}/{len(tokens)} answer tokens")])

    return Task(f"longmemeval:{qid}", str(entry.get("question_type") or "longmemeval"), build, run)


def longmemeval_available(path: str | os.PathLike) -> bool:
    """True when a LongMemEval JSON file exists at `path` (nothing is downloaded).
    `path` 处存在 LongMemEval JSON 文件时为真（不会下载）。

    Input / 输入：
        path: File path.
        path：文件路径。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    return pathlib.Path(path).is_file()


def longmemeval_tasks(path: str | os.PathLike, *, limit: int | None = None, top_k: int = 5) -> list[Task]:
    """Load a LongMemEval JSON file into Tasks; an absent file yields [] so callers can skip.
    将 LongMemEval JSON 文件加载为 Task；文件缺失时返回 []，便于调用方跳过。

    Input / 输入：
        path: Path to longmemeval_*.json (a list of entries).
        path：longmemeval_*.json 的路径（条目列表）。

        limit: Keep only the first `limit` entries.
        limit：只保留前 `limit` 条。

        top_k: Search rows consulted per question.
        top_k：每个问题使用的搜索行数。

    Output / 输出：
        List of Task.
        Task 列表。

    State effect / 状态影响：
        None (reads the file).
        无（读取文件）。
    """
    if not longmemeval_available(path):
        return []
    entries = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        return []
    chosen = entries[:limit] if limit is not None else entries
    return [task_from_longmemeval(e, top_k=top_k) for e in chosen if isinstance(e, dict)]


__all__ = [
    "BLACK",
    "CORRECTION",
    "DC_TEXT",
    "EP1_A",
    "EP2_A",
    "FIXTURES_DIR",
    "LATTE",
    "S4_QUERY",
    "SKILL_TEXT",
    "TASKS",
    "TASK_BY_NAME",
    "UC_TEXT",
    "build_conflict_revision",
    "build_evidence_audit",
    "build_handoff",
    "build_multisession",
    "build_preference_recall",
    "build_rollback",
    "build_skill_distillation",
    "build_world_model",
    "load_fixture",
    "longmemeval_available",
    "longmemeval_tasks",
    "run_conflict_revision",
    "run_evidence_audit",
    "run_handoff",
    "run_multi_session",
    "run_preference_recall",
    "run_rollback",
    "run_skill_distillation",
    "run_temporal",
    "run_world_model",
    "task_from_longmemeval",
]
