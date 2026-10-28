"""Hierarchical coverage compiler (SCC) and PC-IR lowering: L3/L2 interfaces first, L1 atoms on demand (HMS §5.1).
分层覆盖编译器（SCC）与其 PC-IR 降低：先用 L3/L2 接口，仅按需下钻 L1 原子（HMS §5.1）。
"""

# Purpose (EN): The `scc.py` of HMS_arch §5.1 — the compile-time form of "a high-level EP is an interface, refine on
#               demand": cover the query's requirements layer by layer (L3_macro skills, L2_meso InterfaceEP
#               invariants, then L1_micro atoms for whatever stayed uncovered) and lower the result to a PC-IR that
#               carries a fifth obligation, requirements_covered.
# 目的（中）：HMS_arch §5.1 的 `scc.py`——"高层 EP 是接口、按需 Refine"在编译期的形态：逐层覆盖查询需求
#             （L3_macro 技能、L2_meso InterfaceEP 不变量，再用 L1_micro 原子覆盖仍未覆盖者），并把结果降低为
#             携带第五条义务 requirements_covered 的 PC-IR。

# Input (EN): A compiled ContextIR (StateCompiler output or hand-built), requirements, a token budget, a layer order.
# 输入（中）：已编译的 ContextIR（StateCompiler 输出或手工构造）、需求、token 预算、层次顺序。

# Output (EN): HierarchicalIR (reduced ContextIR + per-layer reports) and HierarchicalPCIR (PCIR + coverage obligation).
# 输出（中）：HierarchicalIR（缩减后的 ContextIR + 逐层报告）与 HierarchicalPCIR（PCIR + 覆盖义务）。

# State effect (EN): None (read-only over in-memory objects; compile.py is untouched).
# 状态影响（中）：无（只读处理内存对象；不改动 compile.py）。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from pydantic import Field

from hms_elite.compiler.coverage import CoverageSelection, Requirement, extract_requirements, select_for_coverage
from hms_elite.compiler.pc_ir import lower as lower_context_ir
from hms_elite.compiler.render import render_card
from hms_elite.enums import ObligationStatus, Slot, StrEnum
from hms_elite.exceptions import ProofObligationError
from hms_elite.llms.rule import normalize
from hms_elite.models import PCIR, ContextIR, ContextItem, HMSModel

__all__ = [
    "DEFAULT_LAYERS",
    "REQUIREMENTS_COVERED",
    "CoverageObligation",
    "HierarchicalIR",
    "HierarchicalPCIR",
    "Layer",
    "LayerReport",
    "compile_hierarchical",
    "layer_of",
    "lower",
    "lower_ir",
]

REQUIREMENTS_COVERED = "requirements_covered"
_SLOT_NAMES = ("facts", "episodes", "preferences", "constraints", "skills", "audit")
_EVIDENCE_PREFIX = "anc:"
_SUGGESTION = (
    "drill down with compile(mode='deep') or widen the budget so every requirement is covered, "
    "or lower with require_witness=False"
)


class Layer(StrEnum):
    """The three scales of the support graph as compiler layers (values equal the Scale enum's episode scales).
    支持图的三个尺度作为编译层（取值与 Scale 枚举的情节尺度一致）。
    """

    L1_micro = "L1_micro"
    L2_meso = "L2_meso"
    L3_macro = "L3_macro"


# Top-down: macro skills, meso interfaces, micro atoms — the refine order of HMS §3.6.
# 自上而下：宏观技能、中观接口、微观原子——HMS §3.6 的 Refine 顺序。
DEFAULT_LAYERS: tuple[Layer, ...] = (Layer.L3_macro, Layer.L2_meso, Layer.L1_micro)


@dataclass
class LayerReport:
    """What one layer was offered, what it selected, what it covered and what it left uncovered.
    某一层被提供了什么、选了什么、覆盖了什么、留下了什么未覆盖。
    """

    layer: str
    candidates: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    covered: dict[str, list[str]] = field(default_factory=dict)
    uncovered: list[str] = field(default_factory=list)
    used_tokens: int = 0
    # False when the layer was skipped because every requirement was already covered above it.
    # 当上层已覆盖全部需求、本层被跳过时为 False。
    refined: bool = True

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict of the report.
        报告的 JSON 友好字典。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with layer, candidates, selected, covered, uncovered, used_tokens, refined.
            含 layer、candidates、selected、covered、uncovered、used_tokens、refined 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "layer": self.layer,
            "candidates": list(self.candidates),
            "selected": list(self.selected),
            "covered": {k: list(v) for k, v in self.covered.items()},
            "uncovered": list(self.uncovered),
            "used_tokens": self.used_tokens,
            "refined": self.refined,
        }


@dataclass
class HierarchicalIR:
    """A ContextIR reduced by hierarchical coverage plus the requirements and per-layer reports that explain it.
    经分层覆盖缩减的 ContextIR，连同解释它的需求与逐层报告。
    """

    ir: ContextIR
    requirements: list[Requirement] = field(default_factory=list)
    layers: list[LayerReport] = field(default_factory=list)
    covered: dict[str, list[str]] = field(default_factory=dict)
    uncovered: list[Requirement] = field(default_factory=list)
    # Constraint / audit items re-attached because a kept episode's κ names them (governance closure).
    # 因被保留情节的 κ 指名而重新附上的约束 / audit 项（治理闭包）。
    kappa_closure: list[str] = field(default_factory=list)
    used_tokens: int = 0
    budget_tokens: int = 0

    def to_text(self) -> str:
        """Render the reduced ContextIR as a Work State Card.
        把缩减后的 ContextIR 渲染为 Work State Card。

        Input / 输入：
            none.
            无。

        Output / 输出：
            The card text (render_card).
            卡片文本（render_card）。

        State effect / 状态影响：
            None.
            无。
        """
        return render_card(self.ir)

    def report(self) -> dict[str, Any]:
        """JSON-friendly summary of the hierarchy (what ContextIR.retrieval['hierarchy'] carries).
        层次结构的 JSON 友好摘要（即 ContextIR.retrieval['hierarchy'] 所携带的内容）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with layers, requirements, covered, uncovered (req ids), kappa_closure, used_tokens, budget_tokens.
            含 layers、requirements、covered、uncovered（req id）、kappa_closure、used_tokens、budget_tokens 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "layers": [layer.as_dict() for layer in self.layers],
            "requirements": [req.as_dict() for req in self.requirements],
            "covered": {k: list(v) for k, v in self.covered.items()},
            "uncovered": [req.req_id for req in self.uncovered],
            "kappa_closure": list(self.kappa_closure),
            "used_tokens": self.used_tokens,
            "budget_tokens": self.budget_tokens,
        }


class CoverageObligation(HMSModel):
    """The fifth proof obligation: every extracted requirement is covered by a selected item.
    第五条证明义务：每条抽取出的需求都被某个选中项覆盖。
    """

    name: Literal["requirements_covered"] = REQUIREMENTS_COVERED
    status: ObligationStatus
    details: dict[str, Any] = Field(default_factory=dict)


class HierarchicalPCIR(PCIR):
    """PCIR plus the requirements_covered obligation (kept apart because ObligationName is a closed enum).
    PCIR 加上 requirements_covered 义务（因 ObligationName 是封闭枚举而单列）。
    """

    coverage_obligations: list[CoverageObligation] = Field(default_factory=list)

    def ok(self) -> bool:
        """True when neither the four base obligations nor requirements_covered is violated.
        四条基础义务与 requirements_covered 都未被违反时为真。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Boolean.
            布尔值。

        State effect / 状态影响：
            None.
            无。
        """
        base = super().ok()
        return base and all(ob.status != ObligationStatus.violated.value for ob in self.coverage_obligations)

    def all_obligations(self) -> list[dict[str, Any]]:
        """The five obligations as plain dicts, base four first.
        五条义务的普通字典形式，基础四条在前。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of {'name', 'status', 'details'}.
            {'name', 'status', 'details'} 列表。

        State effect / 状态影响：
            None.
            无。
        """
        rows = [ob.model_dump() for ob in self.proof_obligations]
        rows.extend(ob.model_dump() for ob in self.coverage_obligations)
        for row in rows:
            row["name"] = str(getattr(row["name"], "value", row["name"]))
            row["status"] = str(getattr(row["status"], "value", row["status"]))
        return rows


def layer_of(item: ContextItem) -> Layer:
    """The compiler layer of an item: L3_macro / L2_meso by scale_level, everything else (state, event, L1...) is L1.
    项所属的编译层：按 scale_level 判 L3_macro / L2_meso，其余（state、event、L1……）归 L1。

    Input / 输入：
        item: A ContextItem.
        item：ContextItem。

    Output / 输出：
        Layer.
        Layer。

    State effect / 状态影响：
        None.
        无。
    """
    scale = str(getattr(item.scale_level, "value", item.scale_level) or "")
    if scale == Layer.L3_macro.value:
        return Layer.L3_macro
    if scale == Layer.L2_meso.value:
        return Layer.L2_meso
    return Layer.L1_micro


def _is_evidence(item: ContextItem) -> bool:
    """True for the 'anc:<anchor_id>' evidence rows of the audit slot.
    audit 槽位中 'anc:<anchor_id>' 证据行为真。
    """
    return str(item.slot) == Slot.audit.value and item.item_id.startswith(_EVIDENCE_PREFIX)


def _kappa_closure(ir: ContextIR, keep: list[ContextItem]) -> list[ContextItem]:
    """Constraint / non-evidence audit items of `ir` that spell a κ constraint of a kept episode and are not kept yet.
    `ir` 中拼写出某个被保留情节的 κ 约束、且尚未保留的约束 / 非证据 audit 项。
    """
    kept_ids = {item.item_id for item in keep}
    wanted = {
        normalize(str(text))
        for item in keep
        if str(item.slot) == Slot.episodes.value
        for text in item.explain.get("kappa") or []
    }
    if not wanted:
        return []
    out: list[ContextItem] = []
    for item in list(ir.constraints) + [a for a in ir.audit if not _is_evidence(a)]:
        if item.item_id not in kept_ids and normalize(item.text) in wanted:
            out.append(item)
            kept_ids.add(item.item_id)
    return out


def _layers_of(layers: Iterable[Layer | str] | None) -> list[Layer]:
    """Validate and de-duplicate the requested layer order (None → DEFAULT_LAYERS).
    校验并去重请求的层次顺序（None → DEFAULT_LAYERS）。
    """
    if layers is None:
        return list(DEFAULT_LAYERS)
    out: list[Layer] = []
    for layer in layers:
        value = Layer(str(getattr(layer, "value", layer)))
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("layers must name at least one Layer")
    return out


def _reduced_ir(ir: ContextIR, keep: set[str], *, used: int, budget: int, hierarchy: dict[str, Any]) -> ContextIR:
    """Copy of `ir` restricted to `keep` item ids (original order), with the hierarchy report in retrieval.
    只保留 `keep` 项 id（保持原顺序）的 `ir` 副本，并把层次报告写入 retrieval。
    """
    update: dict[str, Any] = {name: [i for i in getattr(ir, name) if i.item_id in keep] for name in _SLOT_NAMES}
    update["used_tokens"] = used
    update["budget_tokens"] = budget
    update["retrieval"] = {**ir.retrieval, "hierarchy": hierarchy}
    return ir.model_copy(update=update)


def compile_hierarchical(
    ir: ContextIR,
    *,
    requirements: list[Requirement] | None = None,
    budget_tokens: int | None = None,
    layers: Iterable[Layer | str] | None = None,
) -> HierarchicalIR:
    """Cover the requirements layer by layer: higher layers first, lower layers only for what stayed uncovered.
    逐层覆盖需求：先高层，低层只处理仍未覆盖者。

    Input / 输入：
        ir: A compiled ContextIR (its items carry scale_level; 'anc:' evidence rows are re-attached to the selection).
        ir：已编译的 ContextIR（各项携带 scale_level；'anc:' 证据行会重新附到选择结果上）。

        requirements: Requirements to cover (None → extract_requirements(ir.query, ir.retrieval['plan'], ir.as_of)).
        requirements：待覆盖需求（None → extract_requirements(ir.query, ir.retrieval['plan'], as_of=ir.as_of)）。

        budget_tokens: Token budget shared by all layers (None → ir.budget_tokens).
        budget_tokens：所有层共享的 token 预算（None → ir.budget_tokens）。

        layers: Layer order (None → L3_macro, L2_meso, L1_micro); a layer with nothing left to cover is skipped.
        layers：层次顺序（None → L3_macro、L2_meso、L1_micro）；无剩余需求的层被跳过。

    Output / 输出：
        HierarchicalIR whose ir keeps only the selected items, the κ constraints of kept episodes and the evidence rows
        they cite.
        HierarchicalIR，其 ir 只保留选中项、被保留情节的 κ 约束及其引用的证据行。

    State effect / 状态影响：
        None.
        无。
    """
    reqs = (
        list(requirements)
        if requirements is not None
        else extract_requirements(ir.query, ir.retrieval.get("plan") or None, as_of=ir.as_of)
    )
    budget = int(budget_tokens or ir.budget_tokens)
    if budget <= 0:
        raise ValueError("budget_tokens must be positive")
    order = _layers_of(layers)
    pool = [item for item in ir.items() if not _is_evidence(item)]
    by_layer: dict[Layer, list[ContextItem]] = {layer: [] for layer in Layer}
    for item in pool:
        by_layer[layer_of(item)].append(item)
    remaining: list[Requirement] = list(reqs)
    keep: list[ContextItem] = []
    reports: list[LayerReport] = []
    used = 0
    for layer in order:
        candidates = by_layer[layer]
        report = LayerReport(layer=layer.value, candidates=[i.item_id for i in candidates])
        if not remaining or not candidates:
            # Nothing left to cover → the layer is not refined; nothing offered → consulted but empty.
            # 没有剩余需求 → 不下钻本层；没有候选 → 已查询但为空。
            report.refined = bool(remaining)
            report.uncovered = [req.req_id for req in remaining]
            reports.append(report)
            continue
        left = max(1, budget - used)
        sel: CoverageSelection = select_for_coverage(candidates, remaining, budget_tokens=left, min_per_slot=0)
        picked = [item for item in sel.selected if not _is_evidence(item)]
        keep.extend(picked)
        used += sum(item.tokens for item in picked)
        report.selected = [item.item_id for item in picked]
        report.covered = dict(sel.covered)
        remaining = list(sel.uncovered)
        report.uncovered = [req.req_id for req in remaining]
        report.used_tokens = sum(item.tokens for item in picked)
        reports.append(report)
    closure = _kappa_closure(ir, keep)
    keep.extend(closure)
    used += sum(item.tokens for item in closure)
    cited = {anchor for item in keep for anchor in item.anchor_ids}
    evidence = [item for item in ir.audit if _is_evidence(item) and item.item_id[len(_EVIDENCE_PREFIX) :] in cited]
    used += sum(item.tokens for item in evidence)
    covered: dict[str, list[str]] = {}
    for report in reports:
        for req_id, item_ids in report.covered.items():
            covered.setdefault(req_id, []).extend(i for i in item_ids if i not in covered.get(req_id, []))
    covered = {req.req_id: covered[req.req_id] for req in reqs if req.req_id in covered}
    hir = HierarchicalIR(
        ir=ir,
        requirements=reqs,
        layers=reports,
        covered=covered,
        uncovered=remaining,
        kappa_closure=[item.item_id for item in closure],
        used_tokens=used,
        budget_tokens=budget,
    )
    keep_ids = {item.item_id for item in keep} | {item.item_id for item in evidence}
    hir.ir = _reduced_ir(ir, keep_ids, used=used, budget=budget, hierarchy=hir.report())
    return hir


def lower(hir: HierarchicalIR, *, require_witness: bool = False) -> HierarchicalPCIR:
    """Lower a HierarchicalIR to PC-IR: the four base obligations plus requirements_covered.
    把 HierarchicalIR 降低为 PC-IR：四条基础义务加 requirements_covered。

    Input / 输入：
        hir: compile_hierarchical output.
        hir：compile_hierarchical 的输出。

        require_witness: Raise ProofObligationError(PCIR_001) on a violated obligation instead of returning ok() False.
        require_witness：任一义务被违反时抛出 ProofObligationError(PCIR_001)，而非返回 ok() False。

    Output / 输出：
        HierarchicalPCIR; requirements_covered is violated (details['uncovered'] lists them) when any requirement is
        uncovered, skipped when there were no requirements, satisfied otherwise.
        HierarchicalPCIR；任一需求未覆盖时 requirements_covered 为 violated（details['uncovered'] 列出），
        没有需求时为 skipped，否则为 satisfied。

    State effect / 状态影响：
        None.
        无。
    """
    base = lower_context_ir(hir.ir, require_witness=False)
    if not hir.requirements:
        status = ObligationStatus.skipped
    elif hir.uncovered:
        status = ObligationStatus.violated
    else:
        status = ObligationStatus.satisfied
    coverage = CoverageObligation(
        status=status,
        details={
            "uncovered": [req.as_dict() for req in hir.uncovered],
            "covered": {k: list(v) for k, v in hir.covered.items()},
            "layers": [layer.as_dict() for layer in hir.layers],
        },
    )
    payload = dict(base.payload)
    payload["requirements"] = {
        "all": [req.as_dict() for req in hir.requirements],
        "covered": {k: list(v) for k, v in hir.covered.items()},
        "uncovered": [req.req_id for req in hir.uncovered],
    }
    pcir = HierarchicalPCIR(
        payload=payload,
        witness=base.witness,
        human_gloss=base.human_gloss,
        proof_obligations=base.proof_obligations,
        coverage_obligations=[coverage],
    )
    if require_witness and not pcir.ok():
        by_name = {str(getattr(c.name, "value", c.name)): c for c in base.proof_obligations}
        violated = [name for name, c in by_name.items() if c.status == ObligationStatus.violated]
        if status == ObligationStatus.violated:
            violated.append(REQUIREMENTS_COVERED)
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
                "uncovered_requirements": [req.req_id for req in hir.uncovered],
            },
            suggestion=_SUGGESTION,
        )
    return pcir


def lower_ir(ir: ContextIR, *, require_witness: bool = True) -> HierarchicalPCIR:
    """Lower a ContextIR compiled with selection='coverage' (its retrieval['hierarchy'] carries the layer report).
    降低一个以 selection='coverage' 编译的 ContextIR（其 retrieval['hierarchy'] 携带层报告）。

    Input / 输入：
        ir: A ContextIR whose retrieval['hierarchy'] is a HierarchicalIR.report() dict (StateCompiler writes it).
        ir：retrieval['hierarchy'] 为 HierarchicalIR.report() 字典的 ContextIR（由 StateCompiler 写入）。

        require_witness: Raise ProofObligationError(PCIR_001) on a violated obligation instead of returning ok() False.
        require_witness：任一义务被违反时抛出 ProofObligationError(PCIR_001)，而非返回 ok() False。

    Output / 输出：
        HierarchicalPCIR with the four base obligations plus requirements_covered (see lower()).
        带四条基础义务加 requirements_covered 的 HierarchicalPCIR（见 lower()）。

    State effect / 状态影响：
        None.
        无。
    """
    report = ir.retrieval.get("hierarchy") or {}
    requirements = [
        Requirement(
            req_id=str(row["req_id"]),
            kind=str(row["kind"]),
            text=str(row.get("text") or ""),
            weight=float(row.get("weight") or 0.0),
            tokens=tuple(str(tok) for tok in row.get("tokens") or ()),
            window=tuple(row["window"]) if row.get("window") else None,
        )
        for row in report.get("requirements") or []
        if isinstance(row, dict) and row.get("req_id")
    ]
    by_id = {req.req_id: req for req in requirements}
    layers = [
        LayerReport(
            layer=str(row.get("layer")),
            candidates=[str(i) for i in row.get("candidates") or []],
            selected=[str(i) for i in row.get("selected") or []],
            covered={str(k): [str(i) for i in v] for k, v in (row.get("covered") or {}).items()},
            uncovered=[str(i) for i in row.get("uncovered") or []],
            used_tokens=int(row.get("used_tokens") or 0),
            refined=bool(row.get("refined", True)),
        )
        for row in report.get("layers") or []
        if isinstance(row, dict)
    ]
    hir = HierarchicalIR(
        ir=ir,
        requirements=requirements,
        layers=layers,
        covered={str(k): [str(i) for i in v] for k, v in (report.get("covered") or {}).items()},
        uncovered=[by_id[str(i)] for i in report.get("uncovered") or [] if str(i) in by_id],
        kappa_closure=[str(i) for i in report.get("kappa_closure") or []],
        used_tokens=int(report.get("used_tokens") or ir.used_tokens),
        budget_tokens=int(report.get("budget_tokens") or ir.budget_tokens),
    )
    return lower(hir, require_witness=require_witness)
