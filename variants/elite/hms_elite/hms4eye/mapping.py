"""Pure field mapping between HMS Elite rows and the hms4eye v0.6.1 response shapes.
HMS Elite 行与 hms4eye v0.6.1 响应形状之间的纯字段映射。
"""

# Purpose (EN): Project a §4.1 HMS row onto the hms4eye RecallResult / memory-unit dicts and split reserved metadata.
# 目的（中）：把 §4.1 的 HMS 行投影为 hms4eye 的 RecallResult / 记忆单元字典，并拆分保留元数据。

# Input (EN): One public HMS row dict (MemoryAtom.to_public_dict output) or a caller metadata dict.
# 输入（中）：一个公开 HMS 行字典（MemoryAtom.to_public_dict 的输出）或调用方元数据字典。

# Output (EN): Plain dicts / tuples shaped for hms4eye; never HMS models and never partial rows.
# 输出（中）：为 hms4eye 定形的普通字典 / 元组；从不返回 HMS 模型，也不返回残缺的行。

# State effect (EN): None; every function here is pure and copies whatever it reads.
# 状态影响（中）：无；此处每个函数都是纯函数，读取到的内容一律复制。

from __future__ import annotations

from typing import Any

# hms4eye reserves this metadata prefix for its own tenant / bank / document bookkeeping.
# hms4eye 将该元数据前缀保留给它自己的租户 / 记忆库 / 文档簿记使用。
RESERVED_PREFIX = "_hms_"

# The three fact types hms4eye v0.6.1 knows; ExtractRequest only accepts the first two.
# hms4eye v0.6.1 认识的三种事实类型；ExtractRequest 只接受前两种。
FACT_TYPES: tuple[str, str, str] = ("world", "experience", "observation")

# HMS source kinds that hms4eye calls lived experience (an EP is one experience unit).
# hms4eye 称为亲历经验的 HMS 来源种类（一个 EP 就是一个经验单元）。
EXPERIENCE_SOURCE_KINDS = frozenset({"ep"})

# Source kinds that READ BACK as hms4eye's `observation` fact type: consolidation output whose text is
# a statement. `entity` atoms are consolidation output too, but their text is a canonical name rather
# than an observation, so they stay `world` here. capability.CONSOLIDATION_SOURCE_KINDS answers the
# other question -- what clear_observations archives -- and therefore does include `entity`.
# 回读时呈现为 hms4eye `observation` 事实类型的来源种类：文本是陈述句的 consolidation 产物。
# `entity` 原子同样是 consolidation 产物，但其文本是规范名而非观察陈述，因此在此仍归为 `world`。
# capability.CONSOLIDATION_SOURCE_KINDS 回答的是另一个问题——clear_observations 归档什么——所以它包含 `entity`。
OBSERVATION_SOURCE_KINDS = frozenset({"interface_ep", "skill"})

# Default when a row carries no usable source kind; hms4eye's own reader defaults the same way.
# 行中没有可用来源种类时的默认值；hms4eye 自己的读取器也是同样的默认。
DEFAULT_FACT_TYPE = "world"


def _hms_block(row: dict[str, Any]) -> dict[str, Any]:
    """The row's 'hms' sub-dict, or an empty dict when absent or malformed.
    行的 'hms' 子字典；缺失或形状不对时返回空字典。
    """
    block = row.get("hms")
    return dict(block) if isinstance(block, dict) else {}


def _metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    """The row's caller metadata, or an empty dict when it is None (HMS writes None for empty).
    行的调用方元数据；为 None 时返回空字典（HMS 对空元数据写入 None）。
    """
    metadata = row.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _string_list(value: Any) -> list[str]:
    """Coerce a value into a list of strings; a non-list becomes an empty list.
    把一个值规整为字符串列表；非列表规整为空列表。
    """
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _ordered_union(*groups: list[str]) -> list[str]:
    """Concatenate the groups keeping first-occurrence order and dropping later duplicates (§0.1).
    按首次出现顺序拼接各组并丢弃后续重复项（§0.1）。
    """
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return merged


def split_reserved(metadata: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a metadata dict into its hms4eye reserved `_hms_*` keys and the caller's own keys.
    把元数据字典拆分为 hms4eye 的 `_hms_*` 保留键与调用方自己的键。

    Input / 输入：
        metadata: Caller metadata as stored on the row; None is treated as empty.
        metadata：行上存储的调用方元数据；None 视为空。

    Output / 输出：
        (reserved, user) — two new dicts; keys keep their original insertion order within each dict.
        (reserved, user)——两个新字典；每个字典内的键保持原插入顺序。

    State effect / 状态影响：
        None; the input dict is never mutated.
        无；从不修改输入字典。
    """
    source = metadata if isinstance(metadata, dict) else {}
    reserved = {key: value for key, value in source.items() if str(key).startswith(RESERVED_PREFIX)}
    user = {key: value for key, value in source.items() if not str(key).startswith(RESERVED_PREFIX)}
    return reserved, user


def fact_type_of(row: dict[str, Any]) -> str:
    """Decide the hms4eye fact type ('world' | 'experience' | 'observation') of one HMS row.
    判定一条 HMS 行的 hms4eye 事实类型（'world' | 'experience' | 'observation'）。

    Input / 输入：
        row: A public HMS row; an explicit metadata['_hms_type'] always wins over the source kind.
        row：一条公开 HMS 行；显式的 metadata['_hms_type'] 始终优先于来源种类。

    Output / 输出：
        One of FACT_TYPES; source_kind 'ep' → 'experience', 'interface_ep'/'skill' → 'observation', else 'world'.
        FACT_TYPES 之一；来源种类 'ep' → 'experience'，'interface_ep'/'skill' → 'observation'，其余 → 'world'。

    State effect / 状态影响：
        None.
        无。
    """
    declared = _metadata_of(row).get("_hms_type")
    if isinstance(declared, str) and declared:
        return declared
    source_kind = _hms_block(row).get("source_kind")
    if source_kind in EXPERIENCE_SOURCE_KINDS:
        return "experience"
    if source_kind in OBSERVATION_SOURCE_KINDS:
        return "observation"
    return DEFAULT_FACT_TYPE


def to_recall_result(row: dict[str, Any]) -> dict[str, Any]:
    """Project one HMS row onto the 13 fields of the hms4eye RecallResult schema.
    把一条 HMS 行投影为 hms4eye RecallResult 模式的 13 个字段。

    Input / 输入：
        row: A public HMS row as returned by HMS.search() / HMS.get_all() / HMS.get().
        row：HMS.search() / HMS.get_all() / HMS.get() 返回的公开 HMS 行。

    Output / 输出：
        A dict with every RecallResult key; 'metadata' keeps only the caller's non-reserved keys.
        含全部 RecallResult 键的字典；'metadata' 只保留调用方的非保留键。

    State effect / 状态影响：
        None.
        无。
    """
    hms = _hms_block(row)
    reserved, user_metadata = split_reserved(row.get("metadata"))
    # Validity window is the event time HMS resolved, which is what hms4eye means by "occurred".
    # 有效性窗口就是 HMS 解析出的事件时间，也正是 hms4eye 所说的“发生时间”。
    return {
        "id": str(row.get("id", "")),
        "text": str(row.get("memory", "")),
        "type": fact_type_of(row),
        "entities": _string_list(hms.get("entity_ids")),
        "context": reserved.get("_hms_context"),
        "occurred_start": row.get("valid_from"),
        "occurred_end": row.get("valid_to"),
        "mentioned_at": row.get("created_at"),
        "document_id": reserved.get("_hms_document_id"),
        "metadata": user_metadata,
        "chunk_id": None,
        "tags": _ordered_union(_string_list(hms.get("topic_tags")), _string_list(reserved.get("_hms_tags"))),
        # Anchors are the witnesses of this atom: the one field hms4eye's mem0 path can never fill.
        # 锚点是这个原子的见证：hms4eye 的 mem0 路径永远填不上的那个字段。
        "source_fact_ids": _string_list(row.get("anchor_ids")),
    }


def to_memory_unit(row: dict[str, Any]) -> dict[str, Any]:
    """Project one HMS row onto the compact unit hms4eye lists from /memories/list and /memories/{id}.
    把一条 HMS 行投影为 hms4eye 在 /memories/list 与 /memories/{id} 中列出的紧凑单元。

    Input / 输入：
        row: A public HMS row as returned by HMS.get_all() / HMS.get().
        row：HMS.get_all() / HMS.get() 返回的公开 HMS 行。

    Output / 输出：
        {'id', 'text', 'type', 'context', 'date', 'entities'}; 'date' prefers the event time over ingest time.
        {'id', 'text', 'type', 'context', 'date', 'entities'}；'date' 优先取事件时间而非写入时间。

    State effect / 状态影响：
        None.
        无。
    """
    hms = _hms_block(row)
    reserved, _user_metadata = split_reserved(row.get("metadata"))
    return {
        "id": str(row.get("id", "")),
        "text": str(row.get("memory", "")),
        "type": fact_type_of(row),
        "context": reserved.get("_hms_context"),
        "date": row.get("valid_from") or reserved.get("_hms_event_date") or row.get("created_at"),
        "entities": _string_list(hms.get("entity_ids")),
    }


__all__ = [
    "DEFAULT_FACT_TYPE",
    "EXPERIENCE_SOURCE_KINDS",
    "FACT_TYPES",
    "OBSERVATION_SOURCE_KINDS",
    "RESERVED_PREFIX",
    "fact_type_of",
    "split_reserved",
    "to_memory_unit",
    "to_recall_result",
]


def project_reserved(row: dict[str, Any]) -> dict[str, Any]:
    """Copy one HMS row's rich fields into the `_hms_*` metadata namespace hms4eye's own reader consumes.
    把一条 HMS 行的富字段复制进 hms4eye 自己的读取器所消费的 `_hms_*` 元数据命名空间。

    Input / 输入：
        row: A public HMS row; its existing metadata is never overwritten, only completed.
        row：一条公开 HMS 行；其已有的元数据从不被覆盖，只做补全。

    Output / 输出：
        A new metadata dict: the caller's keys plus `_hms_type` / `_hms_event_date` / `_hms_event_end` /
        `_hms_entities` / `_hms_source_fact_ids` / `_hms_confidence` and a widened `_hms_tags`.
        一个新的元数据字典：调用方原有的键，外加 `_hms_type` / `_hms_event_date` / `_hms_event_end` /
        `_hms_entities` / `_hms_source_fact_ids` / `_hms_confidence`，以及被拓宽的 `_hms_tags`。

    State effect / 状态影响：
        None; the input row and its metadata are left untouched.
        无；输入行及其元数据保持不变。
    """
    metadata = dict(_metadata_of(row))
    hms = _hms_block(row)
    # hms4eye's `_to_recall_result` reads `_hms_type` and defaults to 'world'; supply the real one.
    # hms4eye 的 `_to_recall_result` 读取 `_hms_type` 并默认为 'world'；这里补上真实值。
    metadata.setdefault("_hms_type", fact_type_of(row))
    # Its `occurred_start` comes from `_hms_event_date`; our validity interval is the honest source.
    # 它的 `occurred_start` 取自 `_hms_event_date`；我们的有效区间才是诚实的来源。
    event_date = row.get("valid_from") or hms.get("time_anchor")
    if event_date:
        metadata.setdefault("_hms_event_date", event_date)
    # `_hms_event_end` is already in hms4eye's reflect vocabulary; fill it from the closing bound.
    # `_hms_event_end` 已在 hms4eye 的 reflect 词表中；用有效区间的闭端填充。
    if row.get("valid_to"):
        metadata.setdefault("_hms_event_end", row["valid_to"])
    # These two have no hms4eye reader yet, so they ride in metadata until it adopts `to_recall_result`.
    # 这两项目前没有 hms4eye 的读取器，因此先搭载在元数据里，直到它改用 `to_recall_result`。
    entities = _string_list(hms.get("entity_ids"))
    if entities:
        metadata.setdefault("_hms_entities", entities)
    witnesses = _string_list(row.get("anchor_ids"))
    if witnesses:
        metadata.setdefault("_hms_source_fact_ids", witnesses)
    if row.get("confidence") is not None:
        metadata.setdefault("_hms_confidence", row["confidence"])
    # Widen the caller's tags with the engine's own topics, preserving the caller's order first.
    # 用引擎自己的主题拓宽调用方的标签，并优先保留调用方的顺序。
    tags = _ordered_union(_string_list(metadata.get("_hms_tags")), _string_list(hms.get("topic_tags")))
    if tags:
        metadata["_hms_tags"] = tags
    return metadata


def project_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one HMS row whose 'metadata' has been completed by :func:`project_reserved`.
    返回一条 HMS 行的副本，其 'metadata' 已由 :func:`project_reserved` 补全。

    Input / 输入：
        row: A public HMS row.
        row：一条公开 HMS 行。

    Output / 输出：
        A shallow row copy with the projected metadata; every other key is passed through unchanged.
        一份浅拷贝的行，带投影后的元数据；其余每个键原样透传。

    State effect / 状态影响：
        None.
        无。
    """
    projected = dict(row)
    projected["metadata"] = project_reserved(row)
    return projected


def project_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`project_row` to every row of a {'results': [...]} envelope.
    对 {'results': [...]} 信封中的每一行应用 :func:`project_row`。

    Input / 输入：
        envelope: A {'results': [row, ...]} dict as returned by HMS.search() / HMS.get_all().
        envelope：HMS.search() / HMS.get_all() 返回的 {'results': [行, ...]} 字典。

    Output / 输出：
        A new envelope with projected rows; a missing or non-list 'results' is passed through.
        一个带投影行的新信封；缺失或非列表的 'results' 原样透传。

    State effect / 状态影响：
        None.
        无。
    """
    rows = envelope.get("results")
    if not isinstance(rows, list):
        return envelope
    return {**envelope, "results": [project_row(row) for row in rows]}
