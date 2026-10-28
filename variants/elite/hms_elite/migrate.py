"""Migration helper: copy memories out of a Mem0 instance into HMS through the public add() API.
迁移辅助：通过公共 add() API 把 Mem0 实例中的记忆复制到 HMS。
"""

# Purpose (EN): Replace Mem0's history.db import (§0.5 #20) with a get_all() walk re-added as raw message atoms.
# 目的（中）：以遍历 get_all() 并按原始消息原子重新添加的方式，替代 Mem0 的 history.db 导入（§0.5 #20）。

# Input (EN): Any object exposing Mem0's get_all(filters=..., top_k=..., show_expired=...) and a target HMS.
# 输入（中）：任何暴露 Mem0 的 get_all(filters=..., top_k=..., show_expired=...) 的对象，以及目标 HMS。

# Output (EN): {'migrated': n, 'skipped': n, 'ids': {old_id: new_memory_id}}.
# 输出（中）：{'migrated': n, 'skipped': n, 'ids': {old_id: new_memory_id}}。

# State effect (EN): Writes through hms.add(infer=False) only.
# 状态影响（中）：仅通过 hms.add(infer=False) 写入。

from __future__ import annotations

import logging
from typing import Any

from hms_elite.exceptions import HMSError
from hms_elite.memory.main import HMS

logger = logging.getLogger(__name__)

_SCOPE_KEYS = ("user_id", "agent_id", "run_id")


def _rows_of(response: Any) -> list[dict]:
    """Rows from a Mem0 get_all() response: {'results': [...]} or a bare list.
    从 Mem0 的 get_all() 响应取行：{'results': [...]} 或裸列表。
    """
    if isinstance(response, dict):
        rows = response.get("results") or []
    else:
        rows = response or []
    return [row for row in rows if isinstance(row, dict)]


def _walk(memory: Any, *, filters: dict, page_size: int) -> list[dict]:
    """get_all(filters, top_k=page_size) then the same with show_expired=True; rows deduplicated by id.
    先 get_all(filters, top_k=page_size)，再以 show_expired=True 调用；按 id 去重。
    """
    seen: set[str] = set()
    rows: list[dict] = []
    calls: list[dict] = [
        {"filters": filters, "top_k": page_size},
        {"filters": filters, "top_k": page_size, "show_expired": True},
    ]
    for kwargs in calls:
        try:
            response = memory.get_all(**kwargs)
        # 源端不接受 show_expired 参数
        except TypeError as exc:  # the source does not accept show_expired
            logger.warning("from_mem0: get_all(%s) rejected: %s", ", ".join(sorted(kwargs)), exc)
            continue
        for row in _rows_of(response):
            key = str(row.get("id") or "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            rows.append(row)
    return rows


def from_mem0(memory: Any, hms: HMS, *, filters: dict, page_size: int = 100) -> dict:
    """Walk a Mem0 instance's get_all() and re-add every row as a raw (infer=False) memory in HMS.
    遍历 Mem0 实例的 get_all()，并把每一行作为原始（infer=False）记忆重新添加到 HMS。

    Input / 输入：
        memory: The Mem0 Memory (or any object with a compatible get_all()).
        memory：Mem0 的 Memory（或任何带兼容 get_all() 的对象）。

        hms: The target HMS instance.
        hms：目标 HMS 实例。

        filters: Mem0 filter dict passed to get_all (its scope keys are the fallback scope of every row).
        filters：传给 get_all 的 Mem0 过滤字典（其范围键是每行的回退范围）。

        page_size: top_k passed to get_all.
        page_size：传给 get_all 的 top_k。

    Output / 输出：
        {'migrated': n, 'skipped': n, 'ids': {old_id: new_memory_id}}.
        {'migrated': n, 'skipped': n, 'ids': {old_id: new_memory_id}}。

    State effect / 状态影响：
        Writes through hms.add(infer=False, timestamp=created_at, ...) only.
        仅通过 hms.add(infer=False, timestamp=created_at, ...) 写入。
    """
    fallback = {k: filters.get(k) for k in _SCOPE_KEYS if isinstance(filters.get(k), str)}
    migrated = 0
    skipped = 0
    ids: dict[str, str] = {}
    for row in _walk(memory, filters=dict(filters), page_size=int(page_size)):
        text = row.get("memory")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        scope = {k: row.get(k) if isinstance(row.get(k), str) else fallback.get(k) for k in _SCOPE_KEYS}
        try:
            response = hms.add(
                text,
                infer=False,
                timestamp=row.get("created_at"),
                metadata=row.get("metadata") or None,
                user_id=scope["user_id"],
                agent_id=scope["agent_id"],
                run_id=scope["run_id"],
                expiration_date=row.get("expiration_date"),
            )
        except HMSError as exc:
            logger.warning("from_mem0: skipping row %r: %s", row.get("id"), exc)
            skipped += 1
            continue
        results = response.get("results") or []
        if not results:
            skipped += 1
            continue
        migrated += 1
        old_id = row.get("id")
        if old_id is not None:
            ids[str(old_id)] = str(results[0]["id"])
    return {"migrated": migrated, "skipped": skipped, "ids": ids}


__all__ = ["from_mem0"]
