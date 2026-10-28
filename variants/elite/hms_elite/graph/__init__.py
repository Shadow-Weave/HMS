"""Support Graph plane: witnessed typed edges, boundary detection, coarsen/project/refine, explanation.
支持图平面：带 witness 的类型化边、边界检测、Coarsen/Project/Refine、解释。
"""

# Purpose (EN): Inheritance path — re-export the SupportGraph, the three operators, skill promotion and the why() core.
# 目的（中）：继承路径 —— 再导出 SupportGraph、三个算子、技能提升与 why() 核心。

# Input (EN): Node id sets, edge specs.
# 输入（中）：节点集合、边规格。

# Output (EN): Edges, InterfaceEPs, subgraphs.
# 输出（中）：边、InterfaceEP、子图。

# State effect (EN): Writes graph tables only through StateStore.
# 状态影响（中）：仅经 StateStore 写图表。

from hms_elite.graph.explain import explain, render_chain, resolve_target
from hms_elite.graph.operators import coarsen, project, promote_skills, refine, round_trip_check
from hms_elite.graph.support_graph import SupportGraph

__all__ = [
    "SupportGraph",
    "coarsen",
    "explain",
    "project",
    "promote_skills",
    "refine",
    "render_chain",
    "resolve_target",
    "round_trip_check",
]
