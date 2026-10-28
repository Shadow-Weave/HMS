"""MemoryBase: the abstract Mem0 memory contract HMS implements.
MemoryBase：HMS 实现的 Mem0 抽象记忆契约。
"""

# Purpose (EN): Pin the five abstract Mem0 methods so signature parity is checkable.
# 目的（中）：固定五个 Mem0 抽象方法，使签名一致性可被检查。

# Input (EN): Nothing at import time.
# 输入（中）：导入时无输入。

# Output (EN): The MemoryBase ABC.
# 输出（中）：MemoryBase 抽象基类。

# State effect (EN): None.
# 状态影响（中）：无。

from __future__ import annotations

from abc import ABC, abstractmethod

from hms_elite.memory.utils import _UNSET


class MemoryBase(ABC):
    """Abstract Mem0 memory interface (get / get_all / update / delete / history).
    抽象的 Mem0 记忆接口（get / get_all / update / delete / history）。
    """

    @abstractmethod
    def get(self, memory_id):
        """Return one memory row by lineage id, or None.
        按谱系 id 返回一条记忆行，不存在时为 None。

        Input / 输入：
            memory_id: Mem0 memory id (lineage id).
            memory_id：Mem0 记忆 id（谱系 id）。

        Output / 输出：
            Row dict or None.
            行字典或 None。

        State effect / 状态影响：
            None.
            无。
        """

    @abstractmethod
    def get_all(self, *, filters=None, top_k=20, show_expired=False, **kwargs):
        """List memories matching filters.
        列出匹配过滤条件的记忆。

        Input / 输入：
            filters: Mem0 filter dict (scope ids inside).
            filters：Mem0 过滤字典（范围 id 在其中）。

            top_k: Result cap.
            top_k：结果上限。

            show_expired: Include expired/archived rows.
            show_expired：包含已过期/归档的行。

            kwargs: Provider-specific extras.
            kwargs：提供者特定的额外参数。

        Output / 输出：
            {'results': [...]} envelope.
            {'results': [...]} 信封。

        State effect / 状态影响：
            None.
            无。
        """

    @abstractmethod
    def update(self, memory_id, text=None, metadata=None, expiration_date=_UNSET, data=None):
        """Update text, metadata or expiration of a memory.
        更新记忆的文本、元数据或过期日期。

        Input / 输入：
            memory_id: Mem0 memory id.
            memory_id：Mem0 记忆 id。

            text: New text (creates a new version).
            text：新文本（创建新版本）。

            metadata: Metadata patch.
            metadata：元数据补丁。

            expiration_date: New expiration; _UNSET = not provided, None = clear.
            expiration_date：新过期日期；_UNSET 表示未提供，None 表示清除。

            data: Deprecated alias of text.
            data：text 的弃用别名。

        Output / 输出：
            {'message': ...} envelope.
            {'message': ...} 信封。

        State effect / 状态影响：
            Writes the store.
            写入存储。
        """

    @abstractmethod
    def delete(self, memory_id):
        """Archive a memory (revision, never deletion).
        归档记忆（修订而非删除）。

        Input / 输入：
            memory_id: Mem0 memory id.
            memory_id：Mem0 记忆 id。

        Output / 输出：
            {'message': ...} envelope.
            {'message': ...} 信封。

        State effect / 状态影响：
            Writes the store.
            写入存储。
        """

    @abstractmethod
    def history(self, memory_id):
        """Return the history rows of a lineage.
        返回谱系的历史行。

        Input / 输入：
            memory_id: Mem0 memory id.
            memory_id：Mem0 记忆 id。

        Output / 输出：
            Bare list of history rows.
            历史行的裸列表。

        State effect / 状态影响：
            None.
            无。
        """


__all__ = ["MemoryBase"]
