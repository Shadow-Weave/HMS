"""Lower a ContextIR to the proof-carrying PC-IR and evaluate its four obligations.
将 ContextIR 降低为携带证明的 PC-IR 并评估其四条义务。
"""

# Purpose (EN): Build payload / witness / human_gloss and check evidence_coverage, no_unresolved_conflict,
#               governance and not_revoked; raise ProofObligationError when required.
# 目的（中）：构造 payload / witness / human_gloss 并检查 evidence_coverage、no_unresolved_conflict、
#             governance 与 not_revoked；必要时抛出 ProofObligationError。

# Input (EN): A ContextIR whose retrieval['anchors'] carries the anchor rows it cites.
# 输入（中）：retrieval['anchors'] 携带所引用锚点行的 ContextIR。

# Output (EN): A PCIR (or ProofObligationError).
# 输出（中）：PCIR（或 ProofObligationError）。

# State effect (EN): None; store-free.
# 状态影响（中）：无；不依赖存储。

from __future__ import annotations

from typing import Any

from hms_elite.compiler.render import render_card
from hms_elite.enums import ObligationName, ObligationStatus, rank
from hms_elite.exceptions import ProofObligationError
from hms_elite.llms.rule import normalize
from hms_elite.models import PCIR, ContextIR, ContextItem, ProofObligation

__all__ = ["ProofObligationError", "lower", "obligations"]

_SUGGESTION = (
    "call why(<edge_id>) and feedback(..., 'state_wrong'|'state_outdated') to resolve, "
    "or lower with require_witness=False"
)
_NON_AUDIT_SLOTS = ("facts", "episodes", "preferences", "constraints", "skills")


def _undisputed(item: ContextItem) -> list[str]:
    """item.anchor_ids minus explain['disputed_anchor_ids'].
    item.anchor_ids 减去 explain['disputed_anchor_ids']。
    """
    disputed = set(item.explain.get("disputed_anchor_ids") or [])
    return [a for a in item.anchor_ids if a not in disputed]


def _anchor_rows(ir: ContextIR) -> dict[str, dict[str, Any]]:
    """anchor_id → anchor dict from retrieval['anchors'].
    由 retrieval['anchors'] 得到 anchor_id → 锚点字典。
    """
    rows: dict[str, dict[str, Any]] = {}
    for row in ir.retrieval.get("anchors") or []:
        if isinstance(row, dict) and row.get("anchor_id"):
            rows[str(row["anchor_id"])] = row
    return rows


def _entry(item: ContextItem) -> dict[str, Any]:
    """The payload entry shared by every slot.
    各槽位共用的载荷条目。
    """
    return {
        "item_id": item.item_id,
        "text": item.text,
        "anchor_ids": list(item.anchor_ids),
        "atom_id": item.atom_id,
        "memory_id": item.memory_id,
    }


def _non_audit_items(ir: ContextIR) -> list[ContextItem]:
    """Items of the five non-audit slots in slot order.
    五个非 audit 槽位的项，按槽位顺序。
    """
    return [item for name in _NON_AUDIT_SLOTS for item in getattr(ir, name)]


def obligations(ir: ContextIR) -> list[ProofObligation]:
    """Evaluate the four proof obligations of a ContextIR (§15.4) without raising.
    评估 ContextIR 的四条证明义务（§15.4），不抛出异常。

    Input / 输入：
        ir: The ContextIR (retrieval['anchors'] and retrieval['audience'] are consulted).
        ir：ContextIR（参考 retrieval['anchors'] 与 retrieval['audience']）。

    Output / 输出：
        [evidence_coverage, no_unresolved_conflict, governance, not_revoked] with status and details.
        [evidence_coverage, no_unresolved_conflict, governance, not_revoked]，含状态与细节。

    State effect / 状态影响：
        None.
        无。
    """
    anchors = _anchor_rows(ir)
    items = _non_audit_items(ir)
    audience = ir.retrieval.get("audience")

    # evidence_coverage: every non-audit item keeps >= 1 undisputed anchor present in retrieval['anchors'].
    # evidence_coverage：每个非 audit 项都保留 >= 1 个出现在 retrieval['anchors'] 中的无争议锚点。
    missing: list[str] = []
    for item in items:
        undisputed = _undisputed(item)
        if not undisputed or not any(a in anchors for a in undisputed):
            missing.append(item.item_id)
    coverage = ProofObligation(
        name=ObligationName.evidence_coverage,
        status=ObligationStatus.violated if missing else ObligationStatus.satisfied,
        details={"missing_witness": missing},
    )

    # no_unresolved_conflict.
    # no_unresolved_conflict（无未解决冲突）。
    conflicts = list(ir.unresolved_conflicts)
    conflict = ProofObligation(
        name=ObligationName.no_unresolved_conflict,
        status=ObligationStatus.violated if conflicts else ObligationStatus.satisfied,
        details={"unresolved_conflicts": conflicts},
    )

    # governance: skipped without an audience.
    # governance：没有 audience 时跳过。
    if audience is None:
        governance = ProofObligation(
            name=ObligationName.governance, status=ObligationStatus.skipped, details={"governance": []}
        )
    else:
        audience_rank = rank(str(audience))
        offenders: list[str] = []
        present = {normalize(item.text) for item in list(ir.constraints) + list(ir.audit)}
        for item in ir.items():
            if rank(str(item.visibility)) < audience_rank:
                offenders.append(item.item_id)
        for item in ir.episodes:
            for text in item.explain.get("kappa") or []:
                if normalize(str(text)) not in present and item.item_id not in offenders:
                    offenders.append(item.item_id)
        for item in items:
            for anchor_id in item.anchor_ids:
                row = anchors.get(anchor_id)
                if row is None:
                    continue
                if rank(str(row.get("permission_scope") or "private")) < audience_rank:
                    if item.item_id not in offenders:
                        offenders.append(item.item_id)
        governance = ProofObligation(
            name=ObligationName.governance,
            status=ObligationStatus.violated if offenders else ObligationStatus.satisfied,
            details={"governance": offenders, "audience": str(audience)},
        )

    # not_revoked: no undisputed anchor of a non-audit item carries revoked_at.
    # not_revoked：非 audit 项的无争议锚点都不带 revoked_at。
    revoked: list[str] = []
    for item in items:
        for anchor_id in _undisputed(item):
            row = anchors.get(anchor_id)
            if row is not None and row.get("revoked_at") and anchor_id not in revoked:
                revoked.append(anchor_id)
    not_revoked = ProofObligation(
        name=ObligationName.not_revoked,
        status=ObligationStatus.violated if revoked else ObligationStatus.satisfied,
        details={"revoked": revoked},
    )
    return [coverage, conflict, governance, not_revoked]


def lower(ir: ContextIR, *, require_witness: bool = True) -> PCIR:
    """Lower a ContextIR to PC-IR: payload, witness map, human gloss and the evaluated obligations.
    将 ContextIR 降低为 PC-IR：载荷、见证映射、可读描述与已评估的义务。

    Input / 输入：
        ir: The ContextIR (its retrieval['anchors'] carries the anchor rows).
        ir：ContextIR（其 retrieval['anchors'] 携带锚点行）。

        require_witness: Raise ProofObligationError on any violated obligation instead of returning ok() False.
        require_witness：任一义务被违反时抛出 ProofObligationError，而非返回 ok() False。

    Output / 输出：
        PCIR.
        PCIR。

    State effect / 状态影响：
        None.
        无。
    """
    checks = obligations(ir)
    payload: dict[str, Any] = {
        "goal": ir.goal,
        "facts": [_entry(item) for item in ir.facts],
        "episodes": [
            {
                **_entry(item),
                "r": item.explain.get("r"),
                "a": item.explain.get("a"),
                "y": item.explain.get("y"),
                "u": item.explain.get("u"),
            }
            for item in ir.episodes
        ],
        "preferences": [_entry(item) for item in ir.preferences],
        "constraints": [_entry(item) for item in ir.constraints],
        "skills": [_entry(item) for item in ir.skills],
        "invalidated": [
            {**_entry(item), "superseded_by_text": item.explain.get("superseded_by_text")}
            for item in ir.audit
            if str(item.explain.get("reason") or "") == "revised"
            or str(item.explain.get("reason") or "").startswith("archived:")
        ],
        "next_action": ir.next_action,
        "as_of": ir.as_of,
        "scope": ir.scope.as_filters(),
    }
    witness: dict[str, Any] = {
        "items": {item.item_id: list(item.anchor_ids) for item in ir.items()},
        "anchors": [dict(row) for row in ir.retrieval.get("anchors") or [] if isinstance(row, dict)],
        "_all": ir.anchor_ids(),
    }
    pcir = PCIR(payload=payload, witness=witness, human_gloss=render_card(ir), proof_obligations=checks)
    violated = [str(getattr(c.name, "value", c.name)) for c in checks if c.status == ObligationStatus.violated]
    if require_witness and violated:
        by_name = {str(getattr(c.name, "value", c.name)): c for c in checks}
        raise ProofObligationError(
            f"proof obligations violated: {', '.join(violated)}",
            "PCIR_001",
            details={
                "violated": violated,
                "missing_witness": list(by_name["evidence_coverage"].details.get("missing_witness") or []),
                "unresolved_conflicts": list(
                    by_name["no_unresolved_conflict"].details.get("unresolved_conflicts") or []
                ),
                "revoked": list(by_name["not_revoked"].details.get("revoked") or []),
                "governance": list(by_name["governance"].details.get("governance") or []),
            },
            suggestion=_SUGGESTION,
        )
    return pcir
