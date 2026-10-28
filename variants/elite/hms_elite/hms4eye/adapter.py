"""Drop-in replacement for the hms4eye compatibility shell's Mem0Adapter, backed by the HMS engine.
hms4eye 兼容壳中 Mem0Adapter 的原位替换实现，底层由 HMS 引擎支撑。
"""

# Purpose (EN): Expose the seven-method engine seam hms4eye duck-types, delegating each call to an HMS facade.
# 目的（中）：暴露 hms4eye 以鸭子类型注入的七方法引擎接缝，把每个调用委托给 HMS 门面。

# Input (EN): An HMS instance, an optional telemetry recorder with a `stage(name)` context manager, mem0-shaped args.
# 输入（中）：一个 HMS 实例、一个可选的带 `stage(name)` 上下文管理器的遥测记录器，以及 mem0 形状的参数。

# Output (EN): The same envelopes mem0 returns ({'results': [...]}, rows, history lists), carrying HMS's richer fields.
# 输出（中）：与 mem0 相同的信封（{'results': [...]}、行、历史列表），并携带 HMS 更丰富的字段。

# State effect (EN): Every method writes or reads through the injected HMS; extract() uses a throwaway in-memory HMS.
# 状态影响（中）：每个方法都通过注入的 HMS 读写；extract() 使用一个用完即弃的内存 HMS。

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from hms_elite.configs.base import HMSConfig
from hms_elite.hms4eye.mapping import project_envelope, project_row
from hms_elite.memory.main import HMS

# hms4eye's telemetry names are its metric names; they stay 'mem0.*' unless the caller overrides the prefix.
# hms4eye 的遥测名就是它的指标名；除非调用方覆盖前缀，否则保持 'mem0.*'。
DEFAULT_STAGE_PREFIX = "mem0"

# Scope id of the throwaway store used by extract(); never reaches the real store.
# extract() 所用一次性存储的范围 id；绝不会进入真实存储。
_EXTRACTION_SCOPE = "hms4eye-extract"


class _NullTelemetry:
    """Telemetry stand-in used when hms4eye injects no recorder; every stage is a no-op.
    hms4eye 未注入记录器时使用的遥测占位对象；每个阶段都是空操作。
    """

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Enter and leave a named stage without recording anything.
        进入并离开一个具名阶段，不记录任何内容。

        Input / 输入：
            name: Stage name that a real recorder would time.
            name：真实记录器会为其计时的阶段名。

        Output / 输出：
            A context manager yielding None.
            一个产出 None 的上下文管理器。

        State effect / 状态影响：
            None.
            无。
        """
        del name
        yield


class HMSEngineAdapter:
    """The hms4eye engine seam: seven mem0-shaped methods over an HMS facade.
    hms4eye 的引擎接缝：基于 HMS 门面的七个 mem0 形状方法。
    """

    def __init__(
        self,
        hms: HMS,
        telemetry: Any | None = None,
        *,
        stage_prefix: str = DEFAULT_STAGE_PREFIX,
        project_fields: bool = True,
    ) -> None:
        """Bind the engine, the telemetry recorder and the stage-name prefix.
        绑定引擎、遥测记录器与阶段名前缀。

        Input / 输入：
            hms: The HMS facade every method delegates to.
            hms：每个方法委托到的 HMS 门面。

            telemetry: An object with a `stage(name)` context manager (Lite's TelemetryRecorder, or
                       `types.SimpleNamespace(stage=hms_elite.telemetry.stage)`); None installs a no-op recorder.
            telemetry：带 `stage(name)` 上下文管理器的对象（Lite 的 TelemetryRecorder，或
                       `types.SimpleNamespace(stage=hms_elite.telemetry.stage)`）；None 时安装空操作记录器。

            stage_prefix: Prefix of every stage name; 'mem0' keeps hms4eye's existing metric names.
            stage_prefix：每个阶段名的前缀；'mem0' 可保持 hms4eye 现有的指标名。

            project_fields: Complete `_hms_*` metadata on reads so hms4eye's own mapper sees HMS fields.
            project_fields：读取时补全 `_hms_*` 元数据，使 hms4eye 自己的映射器能看到 HMS 字段。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None; the injected HMS is used as is and is never closed by this adapter.
            无；注入的 HMS 原样使用，本适配器绝不关闭它。
        """
        self.hms = hms
        self.telemetry: Any = _NullTelemetry() if telemetry is None else telemetry
        self.stage_prefix = stage_prefix
        # When true, read paths complete the `_hms_*` namespace so hms4eye's unchanged reader
        # reports the real fact type and validity interval instead of defaulting to 'world'.
        # 为真时，读路径会补全 `_hms_*` 命名空间，使 hms4eye 未修改的读取器报告真实的事实类型与
        # 有效区间，而不是一律回落到 'world'。
        self.project_fields = project_fields

    def _stage(self, name: str) -> str:
        """Full stage name for one method, e.g. 'mem0.add'.
        某个方法的完整阶段名，例如 'mem0.add'。
        """
        return f"{self.stage_prefix}.{name}"

    def _extraction_config(self, prompt: str | None) -> HMSConfig:
        """Copy of the engine config redirected to a throwaway in-memory store, with `prompt` as custom instructions.
        引擎配置的副本，重定向到一次性内存存储，并把 `prompt` 作为自定义指令。
        """
        data = self.hms.config.model_dump(mode="python")
        data["store"] = {**dict(data.get("store") or {}), "path": ":memory:"}
        data["history_db_path"] = None
        if prompt is not None:
            data["custom_instructions"] = prompt
        return HMSConfig(**data)

    def add(
        self,
        messages: Any,
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Retain messages under a scope; hms4eye's reserved `_hms_*` metadata keys round-trip untouched.
        在某个范围下留存消息；hms4eye 的 `_hms_*` 保留元数据键原样往返。

        Input / 输入：
            messages: A string, a {'role','content'} dict, or a list of such dicts.
            messages：字符串、{'role','content'} 字典或其列表。

            user_id: The bank scope id hms4eye derives from (tenant_id, bank_id).
            user_id：hms4eye 由 (tenant_id, bank_id) 推导出的记忆库范围 id。

            metadata: Caller metadata, including hms4eye's `_hms_tenant_id` / `_hms_bank_id` tenant guards.
            metadata：调用方元数据，含 hms4eye 的 `_hms_tenant_id` / `_hms_bank_id` 租户守卫。

            infer: False stores one raw message atom per message without any extraction.
            infer：False 时每条消息存一个原始消息原子，不做任何抽取。

            prompt: Replacement system prompt for the procedural branch of HMS.add().
            prompt：HMS.add() 过程记忆分支的替代系统提示词。

        Output / 输出：
            {'results': [row, ...]}; each row's 'event' is HMS's real decision ('ADD' | 'UPDATE' | 'NONE').
            {'results': [行, ...]}；每行的 'event' 是 HMS 的真实决策（'ADD' | 'UPDATE' | 'NONE'）。

        State effect / 状态影响：
            Writes source events, anchors, atoms, axes, vectors, graph nodes/edges and history to the engine store.
            向引擎存储写入源事件、锚点、原子、轴、向量、图节点/边与历史。
        """
        with self.telemetry.stage(self._stage("add")):
            return self.hms.add(messages, user_id=user_id, metadata=metadata, infer=infer, prompt=prompt)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 20,
        threshold: float = 0.1,
        metadata_filters: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Recall within a scope, merging hms4eye's metadata filter tree into the HMS filter DSL.
        在某个范围内召回，把 hms4eye 的元数据过滤树合并进 HMS 的过滤 DSL。

        Input / 输入：
            query: Non-empty query text.
            query：非空查询文本。

            user_id: The bank scope id; it pins the HMS scope exactly as mem0's `filters['user_id']` did.
            user_id：记忆库范围 id；它固定 HMS 范围，与 mem0 的 `filters['user_id']` 完全一致。

            top_k: Result cap (hms4eye maps its low/mid/high budget onto 10/20/50).
            top_k：结果上限（hms4eye 把 low/mid/high 预算映射为 10/20/50）。

            threshold: Minimum fused score in [0, 1].
            threshold：融合后最低分数，范围 [0, 1]。

            metadata_filters: hms4eye's tag tree, e.g. {'OR': [{'_hms_tags': {'contains': 'work'}}]}.
            metadata_filters：hms4eye 的标签树，例如 {'OR': [{'_hms_tags': {'contains': 'work'}}]}。

            explain: Attach 'score_details' to every row.
            explain：为每行附加 'score_details'。

        Output / 输出：
            {'results': [row + 'score', ...]} in rank order.
            按排名顺序的 {'results': [行 + 'score', ...]}。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        # Same merge order as the mem0 adapter: the caller's tree first, then user_id pins the scope.
        # 与 mem0 适配器相同的合并顺序：先放调用方的过滤树，再由 user_id 固定范围。
        filters = dict(metadata_filters or {})
        filters["user_id"] = user_id
        with self.telemetry.stage(self._stage("search")):
            found = self.hms.search(query, top_k=top_k, threshold=threshold, filters=filters, explain=explain)
        # Complete the reserved namespace so hms4eye's own reader sees the real fact type and interval.
        # 补全保留命名空间，使 hms4eye 自己的读取器看到真实的事实类型与时间区间。
        return project_envelope(found) if self.project_fields else found

    def extract(self, messages: list[dict[str, str]], *, prompt: str | None = None) -> list[str]:
        """Run HMS's own observation/EP extraction on the messages without touching the engine store.
        在这些消息上运行 HMS 自己的观察/EP 抽取，不触碰引擎存储。

        Input / 输入：
            messages: Coerced chat messages, exactly the list hms4eye's ExtractService builds per item.
            messages：规整后的聊天消息，正是 hms4eye 的 ExtractService 为每个条目构造的列表。

            prompt: hms4eye's extraction strategy, passed down as HMS custom instructions.
            prompt：hms4eye 的抽取策略，作为 HMS 的自定义指令传下去。

        Output / 输出：
            Extracted fact texts in extraction order, duplicates dropped.
            按抽取顺序的抽取事实文本，去除重复。

        State effect / 状态影响：
            None on the engine store; a throwaway in-memory HMS is built and closed inside the call.
            对引擎存储无影响；调用内部构建并关闭一个一次性内存 HMS。
        """
        with self.telemetry.stage(self._stage("extract")):
            scratch = HMS(self._extraction_config(prompt))
            try:
                result = scratch.add(list(messages), user_id=_EXTRACTION_SCOPE, infer=True)
            finally:
                scratch.close()
        texts: list[str] = []
        for row in result.get("results", []):
            text = str(row.get("memory") or "")
            if text and text not in texts:
                texts.append(text)
        return texts

    def get_all(self, *, user_id: str, top_k: int | None = None) -> dict[str, Any]:
        """List every live memory of a scope in time order.
        按时间顺序列出某个范围内的全部存活记忆。

        Input / 输入：
            user_id: The bank scope id.
            user_id：记忆库范围 id。

            top_k: Result cap; None keeps the engine default.
            top_k：结果上限；None 时沿用引擎默认值。

        Output / 输出：
            {'results': [row, ...]} without 'score'.
            不带 'score' 的 {'results': [行, ...]}。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        params: dict[str, Any] = {"filters": {"user_id": user_id}}
        if top_k is not None:
            params["top_k"] = top_k
        with self.telemetry.stage(self._stage("get_all")):
            listed = self.hms.get_all(**params)
        return project_envelope(listed) if self.project_fields else listed

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Read the head version of one lineage, or None when it is unknown or deleted.
        读取某个谱系的头版本；未知或已删除时返回 None。

        Input / 输入：
            memory_id: The lineage id hms4eye stored; it is stable across revisions.
            memory_id：hms4eye 存下的谱系 id；跨修订保持稳定。

        Output / 输出：
            The row, whose 'metadata' still carries hms4eye's tenant guards, or None.
            该行（其 'metadata' 仍带着 hms4eye 的租户守卫），或 None。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        with self.telemetry.stage(self._stage("get")):
            row = self.hms.get(memory_id)
        if row is None:
            return None
        return project_row(row) if self.project_fields else row

    def history(self, memory_id: str) -> list[dict[str, Any]] | dict[str, Any]:
        """Read the lineage's history rows in sequence order.
        按序列顺序读取谱系的历史行。

        Input / 输入：
            memory_id: The lineage id; an unknown id yields an empty list.
            memory_id：谱系 id；未知 id 返回空列表。

        Output / 输出：
            A bare list of history dicts, each carrying at least 'memory_id' and 'event'.
            历史字典的裸列表，每条至少带 'memory_id' 与 'event'。

        State effect / 状态影响：
            None (read-only).
            无（只读）。
        """
        with self.telemetry.stage(self._stage("history")):
            return self.hms.history(memory_id)

    def delete_all(self, *, user_id: str) -> Any:
        """Archive every live head of a scope (content, edges and history are kept).
        归档某个范围内的每个存活头版本（内容、边与历史都保留）。

        Input / 输入：
            user_id: The bank scope id hms4eye clears.
            user_id：hms4eye 要清空的记忆库范围 id。

        Output / 输出：
            The engine's confirmation dict.
            引擎返回的确认字典。

        State effect / 状态影响：
            One transaction: a system event plus one archive and one DELETE history row per head.
            一个事务：一条系统事件，外加每个头版本一次归档与一条 DELETE 历史行。
        """
        with self.telemetry.stage(self._stage("delete_all")):
            return self.hms.delete_all(user_id=user_id)


__all__ = ["DEFAULT_STAGE_PREFIX", "HMSEngineAdapter"]
