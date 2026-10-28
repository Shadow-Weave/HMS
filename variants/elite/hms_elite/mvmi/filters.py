"""Mem0 filter DSL → FilterExpr AST, SQL lowering, an in-Python evaluation twin and JSON1-aware planning.
Mem0 过滤 DSL → FilterExpr 语法树、SQL 降级、Python 求值孪生与感知 JSON1 的拆分规划。
"""

# Purpose (EN): Parse the Mem0 filter grammar over the seven axes + scope + metadata into one AST that lowers to SQL
#               and evaluates in Python with identical semantics.
# 目的（中）：把覆盖七个轴 + 范围 + 元数据的 Mem0 过滤语法解析为同一棵语法树，既能降级为 SQL，也能在 Python 中以相同语义求值。

# Input (EN): Filter dicts (Mem0 grammar), store capabilities, a query Scope, hydrated MemoryAtom instances.
# 输入（中）：过滤字典（Mem0 语法）、存储能力、查询 Scope、已水合的 MemoryAtom 实例。

# Output (EN): FilterExpr trees, (where_sql, params, joins) fragments over memory_atoms, booleans, (sql_part, python_part).
# 输出（中）：FilterExpr 树、针对 memory_atoms 的 (where_sql, params, joins) 片段、布尔值、(sql_part, python_part)。

# State effect (EN): None (pure functions; no SQL is executed here).
# 状态影响（中）：无（纯函数；此处不执行任何 SQL）。

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ConfigDict
from pydantic import ValidationError as PydanticValidationError

from hms_elite.enums import VISIBILITY_RANK, rank
from hms_elite.exceptions import ValidationError, unwrap_hms_error
from hms_elite.models import HMSModel, MemoryAtom, Scope
from hms_elite.timeutil import day_window, to_utc_iso
from hms_elite.utils.text import entity_key

AXIS_COLUMNS: dict[str, str] = {
    "time": "time_anchor",
    "time_anchor": "time_anchor",
    "created_at": "created_at",
    "updated_at": "updated_at",
    "valid_from": "valid_from",
    "valid_to": "valid_to",
    "atom_type": "atom_type",
    "scale_level": "scale_level",
    "scale": "scale_level",
    "role": "role",
    "sentiment": "sentiment",
    "confidence": "confidence",
    "memory_type": "memory_type",
    "attributed_to": "attributed_to",
    "actor_id": "actor_id",
    "visibility": "visibility",
    "subject": "subject",
    "predicate": "predicate",
    "obs_type": "obs_type",
    "memory_id": "memory_id",
    "atom_id": "atom_id",
    "hash": "hash",
    "source_kind": "source_kind",
    "extracted_by": "extracted_by",
}
TIME_COLUMNS: set[str] = {"time_anchor", "created_at", "updated_at", "valid_from", "valid_to"}
SET_AXES: dict[str, str] = {
    "entity": "entity",
    "entities": "entity",
    "topic": "topic",
    "topics": "topic",
    "anchor_id": "anchor",
}
SCOPE_KEYS: tuple[str, ...] = ("user_id", "agent_id", "run_id")
OPERATORS: tuple[str, ...] = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains")
LOGICAL_KEYS: tuple[str, ...] = ("AND", "OR", "NOT")
META_PREFIX = "meta."

_SET_FIELDS = frozenset(SET_AXES.values())
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UNSUPPORTED = "Unsupported operator '{op}'. Supported: " + ", ".join(OPERATORS)
_COMPARE_SQL = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
# Set-axis subqueries (CONTRACT §13.1): (base SELECT, value column). The entity base gains scope clauses at lowering.
# 集合轴子查询（CONTRACT §13.1）：(基础 SELECT, 取值列)。entity 的基础语句会在降级时补上范围子句。
_SET_SUBQUERY: dict[str, tuple[str, str]] = {
    "entity": (
        "SELECT 1 FROM atom_entity ae JOIN entities e ON e.entity_id = ae.entity_id "
        "WHERE ae.atom_id = memory_atoms.atom_id",
        "e.normalized_text",
    ),
    "topic": ("SELECT 1 FROM atom_topic tp WHERE tp.atom_id = memory_atoms.atom_id", "tp.topic"),
    "anchor": (
        "SELECT 1 FROM atom_anchor aa WHERE aa.atom_id = memory_atoms.atom_id AND aa.disputed_at IS NULL",
        "aa.anchor_id",
    ),
}


# ---------------------------------------------------------------------------
# AST
# 语法树（AST）
# ---------------------------------------------------------------------------
class FilterExpr(HMSModel):
    """Base of the filter AST; every node is a frozen pydantic model.
    过滤语法树的基类；每个节点都是冻结的 pydantic 模型。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True, validate_default=True)


class Eq(FilterExpr):
    """field == value (None → IS NULL).
    field == value（None → IS NULL）。
    """

    field: str
    value: Any = None


class Ne(FilterExpr):
    """field != value (NULL rows match, Mem0 must_not semantics).
    field != value（NULL 行匹配，Mem0 must_not 语义）。
    """

    field: str
    value: Any = None


class Gt(FilterExpr):
    """field > value.
    field > value。
    """

    field: str
    value: Any = None


class Gte(FilterExpr):
    """field >= value.
    field >= value。
    """

    field: str
    value: Any = None


class Lt(FilterExpr):
    """field < value.
    field < value。
    """

    field: str
    value: Any = None


class Lte(FilterExpr):
    """field <= value.
    field <= value。
    """

    field: str
    value: Any = None


class In(FilterExpr):
    """field ∈ values (set axes: any member matches).
    field ∈ values（集合轴：任一成员匹配即可）。
    """

    field: str
    values: list[Any]


class NotIn(FilterExpr):
    """field ∉ values (set axes: no member matches).
    field ∉ values（集合轴：无成员匹配）。
    """

    field: str
    values: list[Any]


class Contains(FilterExpr):
    """Case-sensitive substring match.
    区分大小写的子串匹配。
    """

    field: str
    value: str


class IContains(FilterExpr):
    """Case-insensitive substring match.
    不区分大小写的子串匹配。
    """

    field: str
    value: str


class Exists(FilterExpr):
    """field IS NOT NULL (set axes: at least one member).
    field IS NOT NULL（集合轴：至少一个成员）。
    """

    field: str


class And(FilterExpr):
    """Conjunction (empty → true).
    合取（空 → 真）。
    """

    items: list[FilterExpr]


class Or(FilterExpr):
    """Disjunction (empty → false).
    析取（空 → 假）。
    """

    items: list[FilterExpr]


class Not(FilterExpr):
    """Negation.
    否定。
    """

    item: FilterExpr


_LEAF_OPS: dict[type, str] = {
    Eq: "eq",
    Ne: "ne",
    Gt: "gt",
    Gte: "gte",
    Lt: "lt",
    Lte: "lte",
    In: "in",
    NotIn: "nin",
    Contains: "contains",
    IContains: "icontains",
    Exists: "exists",
}


# ---------------------------------------------------------------------------
# Private helpers
# 私有辅助函数
# ---------------------------------------------------------------------------
def _bad_operator(op: str) -> ValidationError:
    """The VALIDATION_012 error for an unknown operator.
    未知运算符的 VALIDATION_012 错误。
    """
    return ValidationError(_UNSUPPORTED.format(op=op), "VALIDATION_012", details={"operator": op})


def _is_meta(field: str) -> bool:
    """True for `meta.<key>` fields.
    `meta.<key>` 字段为真。
    """
    return field.startswith(META_PREFIX)


def _meta_path(field: str) -> list[str]:
    """Key segments of a `meta.<k1>.<k2>` field.
    `meta.<k1>.<k2>` 字段的键段。
    """
    return [seg for seg in field[len(META_PREFIX) :].split(".") if seg != ""]


def _json_path_literal(field: str) -> str:
    """SQL string literal of the JSON path for a meta field (`'$."k1"."k2"'`).
    元数据字段的 JSON 路径 SQL 字面量（`'$."k1"."k2"'`）。
    """
    segments = [seg.replace('"', "").replace("'", "''") for seg in _meta_path(field)]
    return "'$" + "".join(f'."{seg}"' for seg in segments) + "'"


def _resolve_field(key: str, *, force_meta: bool, nested: bool) -> str:
    """Field-name resolution: axis column → set axis → scope column (nested only) → meta namespace.
    字段名解析：轴列 → 集合轴 → 范围列（仅嵌套）→ 元数据命名空间。
    """
    if force_meta:
        return key if key.startswith(META_PREFIX) else META_PREFIX + key
    if key.startswith(META_PREFIX):
        return key
    if key in AXIS_COLUMNS:
        return AXIS_COLUMNS[key]
    if key in SET_AXES:
        return SET_AXES[key]
    if nested and key in SCOPE_KEYS:
        return key
    return META_PREFIX + key


def _normalize_value(field: str, op: str, value: Any) -> Any:
    """Type-aware literal normalization: time columns → UTC ISO (date-only lte/lt → end of day), entity → key.
    按类型归一化字面量：时间列 → UTC ISO（仅日期的 lte/lt → 当日末），entity → 键。
    """
    if field in TIME_COLUMNS:
        if value is None:
            return None
        if op in ("lte", "lt") and isinstance(value, str) and _DATE_ONLY_RE.match(value.strip()):
            return day_window(value.strip())[1]
        return to_utc_iso(value)
    if field == "entity":
        return entity_key(str(value))
    if field == "topic":
        return str(value)
    return value


def _leaf(field: str, op: str, value: Any) -> FilterExpr:
    """Build one leaf node for (field, op, value) with normalized literals.
    为 (field, op, value) 构造一个带归一化字面量的叶节点。
    """
    if op in ("in", "nin"):
        raw = list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
        values = [_normalize_value(field, op, v) for v in raw]
        return In(field=field, values=values) if op == "in" else NotIn(field=field, values=values)
    if op in ("contains", "icontains"):
        text = str(value)
        return Contains(field=field, value=text) if op == "contains" else IContains(field=field, value=text)
    normalized = _normalize_value(field, op, value)
    if op == "eq":
        return Eq(field=field, value=normalized)
    if op == "ne":
        return Ne(field=field, value=normalized)
    if op == "gt":
        return Gt(field=field, value=normalized)
    if op == "gte":
        return Gte(field=field, value=normalized)
    if op == "lt":
        return Lt(field=field, value=normalized)
    if op == "lte":
        return Lte(field=field, value=normalized)
    raise _bad_operator(op)


def _visibility_gate(audience: Any) -> FilterExpr:
    """Lattice gate for a scalar `visibility` filter: every level with rank >= rank(audience).
    标量 `visibility` 过滤的格门控：秩 >= rank(audience) 的所有等级。
    """
    name = str(audience)
    if name not in VISIBILITY_RANK:
        raise ValidationError(
            f"Unknown visibility {audience!r}. Supported: {', '.join(VISIBILITY_RANK)}",
            "VALIDATION_012",
            details={"key": "visibility", "value": audience},
        )
    return In(field="visibility", values=[v for v in VISIBILITY_RANK if rank(v) >= rank(name)])


def _combine(items: list[FilterExpr]) -> FilterExpr | None:
    """None / the single node / And(items).
    None / 单个节点 / And(items)。
    """
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return And(items=items)


def _parse_entry(key: str, value: Any, *, force_meta: bool, nested: bool) -> FilterExpr:
    """Parse one `key: value` entry of the Mem0 grammar into a node.
    将 Mem0 语法的一个 `key: value` 条目解析为节点。
    """
    if key in LOGICAL_KEYS:
        raw_items = value if isinstance(value, (list, tuple)) else [value]
        parsed: list[FilterExpr] = []
        for item in raw_items:
            if not isinstance(item, dict):
                raise ValidationError(
                    f"{key} expects a list of filter dicts", "VALIDATION_012", details={"key": key, "value": item}
                )
            node = _parse_dict(item, force_meta=force_meta, nested=True)
            parsed.append(node if node is not None else And(items=[]))
        if key == "AND":
            return And(items=parsed)
        if key == "OR":
            return Or(items=parsed)
        return Not(item=parsed[0]) if len(parsed) == 1 else Not(item=Or(items=parsed))
    field = _resolve_field(key, force_meta=force_meta, nested=nested)
    if isinstance(value, str) and value == "*":
        return Exists(field=field)
    if isinstance(value, dict):
        nodes = [_leaf(field, op, v) if op in OPERATORS else _raise_operator(op) for op, v in value.items()]
        return _combine(nodes) or And(items=[])
    if isinstance(value, (list, tuple, set, frozenset)):
        return _leaf(field, "in", list(value))
    if field == "visibility" and not force_meta:
        return _visibility_gate(value)
    return _leaf(field, "eq", value)


def _raise_operator(op: str) -> FilterExpr:
    """Always raises VALIDATION_012 (used inside comprehensions).
    始终抛出 VALIDATION_012（用于推导式内）。
    """
    raise _bad_operator(op)


def _is_scope_pin(value: Any) -> bool:
    """True when a scope key carries a plain id (pins the Scope); list / operator dict / '*' forms are column filters.
    范围键携带普通 id 时为真（固定 Scope）；列表 / 操作符字典 / '*' 形式则作为列过滤。
    """
    if value is None or isinstance(value, (dict, list, tuple, set, frozenset)):
        return False
    return not (isinstance(value, str) and value == "*")


def _and_clauses(value: Any) -> list[Any]:
    """The clause list of an `AND` value (a bare dict is the one-item list `_parse_entry` already accepts).
    `AND` 取值的子句列表（裸字典即 `_parse_entry` 本就接受的单元素列表）。
    """
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _collect_and_scope(clauses: list[Any], hoisted: dict[str, Any], taken: set[str]) -> list[Any]:
    """Copy-on-write walk of one AND clause list, moving scalar scope pins into `hoisted`.
    对一个 AND 子句列表做写时复制遍历，把标量范围固定值移入 `hoisted`。

    Input / 输入：
        clauses: The clause list of a conjunction (non-dict items are passed through so _parse_entry still raises).
        clauses：合取的子句列表（非字典项原样透传，使 _parse_entry 仍能抛错）。

        hoisted: Accumulator receiving `{scope key: id}`; mutated in place.
        hoisted：接收 `{范围键: id}` 的累加器；就地修改。

        taken: Scope keys already bound (top-level keys plus everything hoisted so far); mutated in place.
        taken：已绑定的范围键（顶层键加上此前已上提的键）；就地修改。

    Output / 输出：
        The rewritten clause list: hoisted keys removed, emptied clauses dropped, caller dicts never mutated.
        重写后的子句列表：已上提的键被移除，被清空的子句被丢弃，调用方的字典绝不被修改。

    State effect / 状态影响：
        Mutates only the `hoisted` and `taken` accumulators owned by the caller.
        只修改调用方持有的 `hoisted` 与 `taken` 累加器。
    """
    kept: list[Any] = []
    for item in clauses:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        rest: dict[str, Any] = {}
        for key, value in item.items():
            if key in SCOPE_KEYS and key not in taken and _is_scope_pin(value):
                hoisted[key] = value
                taken.add(key)
                continue
            # 嵌套在 AND 里的 AND 同样是必然约束：递归上提（AND 满足结合律）
            if key == "AND":  # an AND nested inside an AND is just as necessary: recurse (AND is associative)
                inner = _collect_and_scope(_and_clauses(value), hoisted, taken)
                if inner:
                    rest[key] = inner
                continue
            rest[key] = value
        if rest:
            kept.append(rest)
    return kept


def _hoist_and_scope(data: dict[str, Any]) -> dict[str, Any]:
    """Lift scalar scope keys out of the top-level `AND` chain so Mem0's flattened-AND form still pins the Scope.
    把标量范围键从顶层 `AND` 链中上提，使 Mem0 展平 AND 的写法同样能固定 Scope。

    Input / 输入：
        data: The top-level filter dict (already a shallow copy owned by parse_filters).
        data：顶层过滤字典（已是 parse_filters 持有的浅拷贝）。

    Output / 输出：
        `data` unchanged when nothing is hoistable, else a new dict with the pins at top level and the AND chain
        rewritten (a key already present at top level, and every `OR`/`NOT` branch, is left untouched).
        没有可上提的键时原样返回 `data`，否则返回固定值置于顶层、AND 链已重写的新字典
        （顶层已存在的键以及所有 `OR`/`NOT` 分支保持原样）。

    State effect / 状态影响：
        None (copy-on-write; the caller's filter dict is never mutated).
        无（写时复制；调用方的过滤字典绝不被修改）。
    """
    if "AND" not in data:
        return data
    hoisted: dict[str, Any] = {}
    kept = _collect_and_scope(_and_clauses(data["AND"]), hoisted, set(data))
    if not hoisted:
        return data
    data = {**data, **hoisted}
    if kept:
        data["AND"] = kept
    else:
        del data["AND"]
    return data


def _parse_dict(data: dict, *, force_meta: bool, nested: bool) -> FilterExpr | None:
    """Parse every entry of a dict into an And; top-level scope pins are skipped (parse_filters turns them into Scope).
    将字典中每个条目解析为 And；顶层范围固定值被跳过（由 parse_filters 转为 Scope）。
    """
    nodes: list[FilterExpr] = []
    for key, value in data.items():
        if key in SCOPE_KEYS and not nested and not force_meta:
            if value is None or _is_scope_pin(value):
                continue
            # Mem0 list / operator-dict / '*' form on a scope key: a plain column filter, exactly as inside AND/OR/NOT.
            # 范围键上的 Mem0 列表 / 运算符字典 / '*' 形式：按普通列过滤处理，与 AND/OR/NOT 内部完全一致。
            nodes.append(_parse_entry(str(key), value, force_meta=False, nested=True))
            continue
        nodes.append(_parse_entry(str(key), value, force_meta=force_meta, nested=nested))
    return _combine(nodes)


def _entity_base(scope: Scope) -> tuple[str, list[Any]]:
    """Entity EXISTS base with the identity clauses for every pinned scope id.
    带有各已固定范围 id 身份子句的实体 EXISTS 基础语句。
    """
    base, _ = _SET_SUBQUERY["entity"]
    params: list[Any] = []
    for column in ("user_id", "agent_id"):
        value = getattr(scope, column)
        if value is not None:
            base += f" AND COALESCE(e.{column},'') = COALESCE(?, '')"
            params.append(value)
    return base, params


def _placeholders(n: int) -> str:
    """`?,?,?` for n values.
    n 个值的 `?,?,?`。
    """
    return ",".join("?" for _ in range(n))


def _lower_set(node: FilterExpr, params: list[Any], scope: Scope) -> str:
    """Lower a set-axis leaf (entity / topic / anchor) to an EXISTS subquery.
    将集合轴叶（entity / topic / anchor）降级为 EXISTS 子查询。
    """
    axis = node.field  # type: ignore[attr-defined]
    if axis == "entity":
        base, base_params = _entity_base(scope)
    else:
        base, base_params = _SET_SUBQUERY[axis][0], []
    column = _SET_SUBQUERY[axis][1]
    op = _LEAF_OPS[type(node)]
    if op == "exists":
        params.extend(base_params)
        return f"EXISTS ({base})"
    if op in ("eq", "ne"):
        params.extend(base_params)
        params.append(node.value)  # type: ignore[attr-defined]
        prefix = "EXISTS" if op == "eq" else "NOT EXISTS"
        return f"{prefix} ({base} AND {column} = ?)"
    if op in ("in", "nin"):
        values = list(node.values)  # type: ignore[attr-defined]
        if not values:
            return "0" if op == "in" else "1"
        params.extend(base_params)
        params.extend(values)
        prefix = "EXISTS" if op == "in" else "NOT EXISTS"
        return f"{prefix} ({base} AND {column} IN ({_placeholders(len(values))}))"
    if op == "contains":
        params.extend(base_params)
        params.append(node.value)  # type: ignore[attr-defined]
        return f"EXISTS ({base} AND instr({column}, ?) > 0)"
    if op == "icontains":
        params.extend(base_params)
        params.append(node.value)  # type: ignore[attr-defined]
        return f"EXISTS ({base} AND instr(LOWER({column}), LOWER(?)) > 0)"
    raise ValidationError(
        f"Operator '{op}' is not supported on the '{axis}' axis",
        "VALIDATION_012",
        details={"operator": op, "axis": axis},
    )


def _lower_leaf(node: FilterExpr, column: str, params: list[Any], *, is_meta: bool) -> str:
    """Lower a column / meta leaf given its SQL column expression.
    按给定的 SQL 列表达式降级列 / 元数据叶。
    """
    # Every branch yields 0/1, never SQL NULL, so NOT/OR over a missing value behave exactly like the Python twin
    # 每个分支都产出 0/1，绝不产生 SQL NULL，因此对缺失值取 NOT/OR 的行为与 Python 孪生实现完全一致
    # (SQLite three-valued logic would otherwise drop NULL rows inside NOT (...)).
    # （否则 SQLite 的三值逻辑会在 NOT (...) 内部丢掉 NULL 行）。
    op = _LEAF_OPS[type(node)]
    if op == "exists":
        return f"{column} IS NOT NULL"
    if op in ("in", "nin"):
        values = list(node.values)  # type: ignore[attr-defined]
        has_none = any(v is None for v in values)
        rest = [v for v in values if v is not None]
        if op == "in":
            if not values:
                return "0"
            parts: list[str] = []
            if has_none:
                parts.append(f"{column} IS NULL")
            if rest:
                params.extend(rest)
                parts.append(f"({column} IS NOT NULL AND {column} IN ({_placeholders(len(rest))}))")
            return "(" + " OR ".join(parts) + ")"
        if not values:
            return "1"
        if has_none:
            if not rest:
                return f"{column} IS NOT NULL"
            params.extend(rest)
            return f"({column} IS NOT NULL AND {column} NOT IN ({_placeholders(len(rest))}))"
        params.extend(rest)
        return f"({column} IS NULL OR {column} NOT IN ({_placeholders(len(rest))}))"
    if op == "contains":
        params.append(node.value)  # type: ignore[attr-defined]
        return f"({column} IS NOT NULL AND instr({column}, ?) > 0)"
    if op == "icontains":
        params.append(node.value)  # type: ignore[attr-defined]
        return f"({column} IS NOT NULL AND instr(LOWER({column}), LOWER(?)) > 0)"
    value = node.value  # type: ignore[attr-defined]
    if op == "eq":
        if value is None:
            return f"{column} IS NULL"
        params.append(value)
        return f"({column} IS NOT NULL AND {column} = ?)"
    if op == "ne":
        if value is None:
            return f"{column} IS NOT NULL"
        params.append(value)
        return f"({column} IS NULL OR {column} != ?)"
    sql_op = _COMPARE_SQL[op]
    if value is None:
        return "0"
    if is_meta:
        # json_extract has no column affinity: guard the SQLite type so TEXT never sorts above numbers (Python twin
        # json_extract 没有列亲和性：需守卫 SQLite 类型，使 TEXT 永不排在数字之上（Python 孪生实现在
        # returns False on mixed types); typeof(NULL) = 'null' fails the guard, so missing keys yield 0.
        # 混合类型上返回 False）；typeof(NULL) = 'null' 通不过该守卫，因此缺失的键结果为 0。
        if isinstance(value, bool):
            params.append(int(value))
            return f"(typeof({column}) IN ('integer','real') AND {column} {sql_op} ?)"
        if isinstance(value, (int, float)):
            params.append(value)
            return f"(typeof({column}) IN ('integer','real') AND {column} {sql_op} ?)"
        if isinstance(value, str):
            params.append(value)
            return f"(typeof({column}) = 'text' AND {column} {sql_op} ?)"
    params.append(value)
    return f"({column} IS NOT NULL AND {column} {sql_op} ?)"


def _lower(node: FilterExpr, params: list[Any], capabilities: dict, scope: Scope) -> str:
    """Recursive lowering of a node.
    节点的递归降级。
    """
    if isinstance(node, And):
        if not node.items:
            return "1"
        return "(" + " AND ".join(_lower(item, params, capabilities, scope) for item in node.items) + ")"
    if isinstance(node, Or):
        if not node.items:
            return "0"
        return "(" + " OR ".join(_lower(item, params, capabilities, scope) for item in node.items) + ")"
    if isinstance(node, Not):
        return "NOT (" + _lower(node.item, params, capabilities, scope) + ")"
    field = node.field  # type: ignore[attr-defined]
    if field in _SET_FIELDS:
        return _lower_set(node, params, scope)
    if _is_meta(field):
        if not capabilities.get("json1", True):
            raise ValidationError(
                "metadata filters require the SQLite JSON1 extension (plan_filters moves them to Python)",
                "VALIDATION_012",
                details={"field": field},
            )
        column = f"json_extract(memory_atoms.user_meta_json, {_json_path_literal(field)})"
        return _lower_leaf(node, column, params, is_meta=True)
    return _lower_leaf(node, f"memory_atoms.{field}", params, is_meta=False)


def _meta_get(user_meta: dict, field: str) -> Any:
    """Walk nested dicts along the meta path; missing → None.
    沿元数据路径遍历嵌套字典；缺失 → None。
    """
    current: Any = user_meta
    for segment in _meta_path(field):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _as_text(value: Any) -> str:
    """Text form used by substring operators (mirrors SQLite instr() coercion of json_extract values).
    子串运算符使用的文本形式（镜像 SQLite instr() 对 json_extract 值的转换）。
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _coerce_pair(actual: Any, expected: Any, *, column: bool) -> tuple[Any, Any]:
    """Apply SQLite column-affinity coercion between a column value and a literal (columns only).
    在列值与字面量之间应用 SQLite 列亲和性转换（仅限列）。
    """
    if not column:
        return actual, expected
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual, expected
    if isinstance(actual, (int, float)) and isinstance(expected, str):
        try:
            return actual, float(expected)
        except ValueError:
            return actual, expected
    if isinstance(actual, str) and isinstance(expected, (int, float)):
        return actual, str(expected)
    return actual, expected


def _equals(actual: Any, expected: Any, *, column: bool) -> bool:
    """Equality with NULL semantics and column coercion.
    带 NULL 语义与列转换的相等判断。
    """
    if expected is None or actual is None:
        return expected is None and actual is None
    a, e = _coerce_pair(actual, expected, column=column)
    if isinstance(a, bool) != isinstance(e, bool) and (isinstance(a, bool) or isinstance(e, bool)):
        # SQLite stores booleans as 0/1: True == 1.
        # SQLite 把布尔值存为 0/1：True == 1。
        return int(a) == int(e) if isinstance(a, (bool, int, float)) and isinstance(e, (bool, int, float)) else False
    try:
        return bool(a == e)
    except TypeError:  # pragma: no cover - defensive
        return False


def _ordered(op: str, actual: Any, expected: Any, *, column: bool) -> bool:
    """Ordered comparison; NULL or mixed types → False (matches the SQL lowering).
    有序比较；NULL 或混合类型 → False（与 SQL 降级一致）。
    """
    if actual is None or expected is None:
        return False
    a, e = _coerce_pair(actual, expected, column=column)
    if isinstance(a, bool):
        a = int(a)
    if isinstance(e, bool):
        e = int(e)
    numeric = (int, float)
    if isinstance(a, numeric) and isinstance(e, numeric):
        pass
    elif isinstance(a, str) and isinstance(e, str):
        pass
    else:
        return False
    if op == "gt":
        return a > e
    if op == "gte":
        return a >= e
    if op == "lt":
        return a < e
    return a <= e


def _eval_leaf(node: FilterExpr, actual: Any, *, column: bool) -> bool:
    """Evaluate a column / meta leaf against the atom's value.
    针对原子的值求值列 / 元数据叶。
    """
    op = _LEAF_OPS[type(node)]
    if op == "exists":
        return actual is not None
    if op == "eq":
        return _equals(actual, node.value, column=column)  # type: ignore[attr-defined]
    if op == "ne":
        return not _equals(actual, node.value, column=column)  # type: ignore[attr-defined]
    if op == "in":
        return any(_equals(actual, v, column=column) for v in node.values)  # type: ignore[attr-defined]
    if op == "nin":
        return not any(_equals(actual, v, column=column) for v in node.values)  # type: ignore[attr-defined]
    if op == "contains":
        return actual is not None and node.value in _as_text(actual)  # type: ignore[attr-defined]
    if op == "icontains":
        return actual is not None and node.value.lower() in _as_text(actual).lower()  # type: ignore[attr-defined]
    return _ordered(op, actual, node.value, column=column)  # type: ignore[attr-defined]


def _eval_set(node: FilterExpr, members: set[str]) -> bool:
    """Evaluate a set-axis leaf against the atom's member set.
    针对原子的成员集合求值集合轴叶。
    """
    op = _LEAF_OPS[type(node)]
    if op == "exists":
        return bool(members)
    if op == "eq":
        return node.value in members  # type: ignore[attr-defined]
    if op == "ne":
        return node.value not in members  # type: ignore[attr-defined]
    if op == "in":
        return any(v in members for v in node.values)  # type: ignore[attr-defined]
    if op == "nin":
        return not any(v in members for v in node.values)  # type: ignore[attr-defined]
    if op == "contains":
        return any(node.value in m for m in members)  # type: ignore[attr-defined]
    if op == "icontains":
        needle = node.value.lower()  # type: ignore[attr-defined]
        return any(needle in m.lower() for m in members)
    raise ValidationError(
        f"Operator '{op}' is not supported on the '{node.field}' axis",  # type: ignore[attr-defined]
        "VALIDATION_012",
        details={"operator": op, "axis": node.field},  # type: ignore[attr-defined]
    )


def _column_value(atom: MemoryAtom, column: str) -> Any:
    """Read a memory_atoms column from the model (scope columns live on atom.scope).
    从模型读取 memory_atoms 列（范围列位于 atom.scope）。
    """
    if column in SCOPE_KEYS:
        return getattr(atom.scope, column)
    return getattr(atom, column, None)


def _has_meta(node: FilterExpr) -> bool:
    """True when any leaf below the node lives in the meta namespace.
    节点下任一叶位于元数据命名空间时为真。
    """
    if isinstance(node, (And, Or)):
        return any(_has_meta(item) for item in node.items)
    if isinstance(node, Not):
        return _has_meta(node.item)
    return _is_meta(node.field)  # type: ignore[attr-defined]


def _split(node: FilterExpr) -> tuple[list[FilterExpr], list[FilterExpr]]:
    """Split a node into (sql-able items, python-only items), flattening nested Ands.
    将节点拆分为（可 SQL 的项，仅 Python 的项），并展开嵌套 And。
    """
    if isinstance(node, And):
        sql_items: list[FilterExpr] = []
        py_items: list[FilterExpr] = []
        for item in node.items:
            s, p = _split(item)
            sql_items.extend(s)
            py_items.extend(p)
        return sql_items, py_items
    if _has_meta(node):
        return [], [node]
    return [node], []


# ---------------------------------------------------------------------------
# Public API (CONTRACT §13.1)
# 公开 API（CONTRACT §13.1）
# ---------------------------------------------------------------------------
def parse_filters(
    filters: dict | None, *, require_scope: bool = True, context: str = "search", force_meta: bool = False
) -> tuple[Scope, FilterExpr | None]:
    """Parse a Mem0 filter dict into (Scope, FilterExpr | None) over the seven axes, scope and metadata.
    将 Mem0 过滤字典解析为覆盖七个轴、范围与元数据的 (Scope, FilterExpr | None)。

    Input / 输入：
        filters: Mem0 grammar (`key: scalar | [..] | '*' | {op: v}`, `AND/OR/NOT: [..]`); None → {}.
        filters：Mem0 语法（`key: scalar | [..] | '*' | {op: v}`、`AND/OR/NOT: [..]`）；None → {}。

        require_scope: Raise VALIDATION_001 when no scope id is present.
        require_scope：没有任何范围 id 时抛出 VALIDATION_001。

        context: Operation name recorded in the VALIDATION_001 details.
        context：记录在 VALIDATION_001 details 中的操作名。

        force_meta: Force every leaf (scope keys included) into the `meta.<key>` namespace (AxisQuery.meta_filters).
        force_meta：把每个叶（含范围键）强制放入 `meta.<key>` 命名空间（AxisQuery.meta_filters）。

    Output / 输出：
        (Scope from the scalar scope keys, AST or None); list/dict/'*' scope keys → column leaves, `visibility` → gate.
        （由标量范围键构造的 Scope，语法树或 None）；列表/字典/'*' 形式的范围键 → 列叶节点，`visibility` → 格门控。

        Mem0 flattens `AND` before reading the scope, so a scalar scope key inside the top-level `AND` chain (nested
        `AND`s included) is hoisted to the top level and pins the Scope; the first occurrence wins, a key already at
        top level blocks the hoist, and `OR`/`NOT` branches keep their column leaves (no scope → VALIDATION_001).
        Mem0 在读取 scope 前会展平 `AND`，因此顶层 `AND` 链（含嵌套 `AND`）中的标量范围键会被上提到顶层并固定
        Scope；以首次出现者为准，顶层已定义该键则不上提，`OR`/`NOT` 分支仍保留列叶节点（无 scope → VALIDATION_001）。

    State effect / 状态影响：
        None.
        无。
    """
    if filters is None:
        data: dict[str, Any] = {}
    elif isinstance(filters, dict):
        data = dict(filters)
    else:
        raise ValidationError(
            f"filters must be a dict, got {type(filters).__name__}", "VALIDATION_012", details={"filters": filters}
        )
    if force_meta:
        scope = Scope()
    else:
        data = _hoist_and_scope(data)
        pinned = {k: data[k] for k in SCOPE_KEYS if k in data and _is_scope_pin(data[k])}
        try:
            scope = Scope(**pinned)
        # Scope id 校验器抛出的 VALIDATION_001 会被 pydantic 包装后传出
        except PydanticValidationError as exc:  # the Scope id validator's VALIDATION_001 arrives wrapped by pydantic
            inner = unwrap_hms_error(exc)
            raise (inner or ValidationError(str(exc), "VALIDATION_001", details={"filters": pinned})) from exc
    if require_scope:
        scope.require_any(context=context)
    return scope, _parse_dict(data, force_meta=force_meta, nested=False)


def lower_to_sql(expr: FilterExpr | None, *, capabilities: dict, scope: Scope) -> tuple[str, list[Any], set[str]]:
    """Lower an AST to a WHERE fragment over memory_atoms (set axes as EXISTS subqueries, meta via json_extract).
    将语法树降级为针对 memory_atoms 的 WHERE 片段（集合轴用 EXISTS 子查询，元数据用 json_extract）。

    Input / 输入：
        expr: The AST from parse_filters / plan_filters, or None.
        expr：来自 parse_filters / plan_filters 的语法树，或 None。

        capabilities: store.capabilities; `json1` must be True for meta leaves.
        capabilities：store.capabilities；元数据叶要求 `json1` 为 True。

        scope: Query scope; its pinned user_id / agent_id restrict the entity identity subquery.
        scope：查询范围；其已固定的 user_id / agent_id 限制实体身份子查询。

    Output / 输出：
        (where_sql, params, joins) — joins is always empty (every axis lowers to a correlated EXISTS).
        (where_sql, params, joins)——joins 始终为空（每个轴都降级为相关 EXISTS）。

    State effect / 状态影响：
        None.
        无。
    """
    if expr is None:
        return "", [], set()
    params: list[Any] = []
    sql = _lower(expr, params, capabilities or {}, scope)
    return sql, params, set()


def evaluate(expr: FilterExpr | None, atom: MemoryAtom, *, entity_keys: set[str], anchor_ids: set[str]) -> bool:
    """In-Python twin of lower_to_sql with identical semantics (JSON1 fallback and conformance tests).
    lower_to_sql 的 Python 孪生，语义完全一致（JSON1 回退与一致性测试）。

    Input / 输入：
        expr: The AST or None (None → True).
        expr：语法树或 None（None → True）。

        atom: A hydrated MemoryAtom.
        atom：已水合的 MemoryAtom。

        entity_keys: The atom's entity keys (entities.normalized_text).
        entity_keys：原子的实体键（entities.normalized_text）。

        anchor_ids: The atom's undisputed anchor ids.
        anchor_ids：原子的无争议锚点 id。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    if expr is None:
        return True
    if isinstance(expr, And):
        return all(evaluate(item, atom, entity_keys=entity_keys, anchor_ids=anchor_ids) for item in expr.items)
    if isinstance(expr, Or):
        return any(evaluate(item, atom, entity_keys=entity_keys, anchor_ids=anchor_ids) for item in expr.items)
    if isinstance(expr, Not):
        return not evaluate(expr.item, atom, entity_keys=entity_keys, anchor_ids=anchor_ids)
    field = expr.field  # type: ignore[attr-defined]
    if field == "entity":
        return _eval_set(expr, set(entity_keys))
    if field == "topic":
        return _eval_set(expr, set(atom.topic_tags))
    if field == "anchor":
        return _eval_set(expr, set(anchor_ids))
    if _is_meta(field):
        return _eval_leaf(expr, _meta_get(atom.user_meta or {}, field), column=False)
    return _eval_leaf(expr, _column_value(atom, field), column=True)


def plan_filters(expr: FilterExpr | None, *, capabilities: dict) -> tuple[FilterExpr | None, FilterExpr | None]:
    """Split an AST into (sql_part, python_part): without JSON1 every meta leaf (and any Or/Not holding one) runs in Python.
    将语法树拆分为 (sql_part, python_part)：没有 JSON1 时，每个元数据叶（及含它的 Or/Not）在 Python 中求值。

    Input / 输入：
        expr: The AST or None.
        expr：语法树或 None。

        capabilities: store.capabilities (`json1` flag).
        capabilities：store.capabilities（`json1` 标志）。

    Output / 输出：
        (sql_part, python_part); python_part is None when JSON1 is available.
        (sql_part, python_part)；JSON1 可用时 python_part 为 None。

    State effect / 状态影响：
        None.
        无。
    """
    if expr is None:
        return None, None
    if (capabilities or {}).get("json1", True):
        return expr, None
    sql_items, py_items = _split(expr)
    return _combine(sql_items), _combine(py_items)


def has_advanced_operators(filters: dict) -> bool:
    """Mem0 port: True when the dict uses AND/OR/NOT, an operator dict or the '*' wildcard.
    Mem0 移植：字典使用 AND/OR/NOT、运算符字典或 '*' 通配符时为真。

    Input / 输入：
        filters: A Mem0 filter dict.
        filters：Mem0 过滤字典。

    Output / 输出：
        Boolean.
        布尔值。

    State effect / 状态影响：
        None.
        无。
    """
    for key, value in (filters or {}).items():
        if key in LOGICAL_KEYS:
            return True
        if isinstance(value, str) and value == "*":
            return True
        if isinstance(value, dict) and any(op in OPERATORS for op in value):
            return True
    return False


__all__ = [
    "AXIS_COLUMNS",
    "LOGICAL_KEYS",
    "META_PREFIX",
    "OPERATORS",
    "SCOPE_KEYS",
    "SET_AXES",
    "TIME_COLUMNS",
    "And",
    "Contains",
    "Eq",
    "Exists",
    "FilterExpr",
    "Gt",
    "Gte",
    "IContains",
    "In",
    "Lt",
    "Lte",
    "Ne",
    "Not",
    "NotIn",
    "Or",
    "evaluate",
    "has_advanced_operators",
    "lower_to_sql",
    "parse_filters",
    "plan_filters",
]
