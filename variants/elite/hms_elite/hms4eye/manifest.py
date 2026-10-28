"""Our side of the hms4eye v0.6.1 capability matrix, stated against the upstream compatibility report.
我们这一侧的 hms4eye v0.6.1 能力矩阵，对照上游兼容性报告给出。
"""

# Purpose (EN): Record, per hms4eye HTTP operationId, what the HMS engine really offers, and quantify the lift
#               over the upstream Mem0 shell whose 33 HTTP no-ops are declared "no equivalent capability".
# 目的（中）：逐个记录 hms4eye HTTP operationId 上 HMS 引擎真正能提供什么，并量化相对上游 Mem0 兼容壳的提升，
#            上游那 33 个 HTTP no-op 被声明为"没有等价能力"。

# Input (EN): Nothing at import time; compare_with_upstream() reads only the two module-level tables.
# 输入（中）：导入时不接收输入；compare_with_upstream() 只读取本模块的两张表。

# Output (EN): CAPABILITY / UPSTREAM / SURFACE tables and the compare_with_upstream() statistics dict.
# 输出（中）：CAPABILITY / UPSTREAM / SURFACE 三张表，以及 compare_with_upstream() 的统计字典。

# State effect (EN): None; this module is pure data plus one pure function and never touches a store.
# 状态影响（中）：无；本模块是纯数据加一个纯函数，从不接触存储。

from __future__ import annotations

# The three verdicts we use. "native" = the engine already models it; "adapted" = the engine models the same
# facts under different names and the capability layer reshapes them; "unsupported" = out of engine scope.
# 我们使用的三种判定。"native" = 引擎本就建模了它；"adapted" = 引擎以不同名字建模同样的事实，由能力层重塑；
# "unsupported" = 不在引擎范围内。
VERDICTS: tuple[str, str, str] = ("native", "adapted", "unsupported")

# Upstream's own verdict per operationId, transcribed from HMS4EYE_INTERFACE_COMPATIBILITY.md (68 HTTP rows,
# 6 native / 29 adapted / 33 no-op). Kept here so the comparison needs no file at runtime.
# 上游对每个 operationId 的判定，抄自 HMS4EYE_INTERFACE_COMPATIBILITY.md（68 个 HTTP 行，
# 6 个 native / 29 个 adapted / 33 个 no-op）。放在这里，使比较在运行时不依赖任何文件。
UPSTREAM: dict[str, str] = {
    "add_bank_background": "adapted",
    "audit_log_stats": "adapted",
    "cancel_operation": "adapted",
    "clear_bank_memories": "native",
    "clear_memory_observations": "no-op",
    "clear_observations": "no-op",
    "create_directive": "no-op",
    "create_mental_model": "no-op",
    "create_or_update_bank": "adapted",
    "create_webhook": "no-op",
    "delete_bank": "adapted",
    "delete_directive": "no-op",
    "delete_document": "adapted",
    "delete_mental_model": "no-op",
    "delete_webhook": "no-op",
    "export_bank_template": "no-op",
    "extract_memories": "adapted",
    "file_retain": "adapted",
    "get_agent_stats": "adapted",
    "get_bank_config": "adapted",
    "get_bank_profile": "adapted",
    "get_bank_template_schema": "no-op",
    "get_chunk": "no-op",
    "get_directive": "no-op",
    "get_document": "adapted",
    "get_entity": "no-op",
    "get_entity_graph": "no-op",
    "get_graph": "no-op",
    "get_memories_timeseries": "adapted",
    "get_memory": "native",
    "get_mental_model": "no-op",
    "get_mental_model_history": "no-op",
    "get_observation_history": "native",
    "get_operation_status": "adapted",
    "get_version": "adapted",
    "health_endpoint_health_get": "adapted",
    "import_bank_template": "no-op",
    "list_audit_logs": "adapted",
    "list_banks": "adapted",
    "list_directives": "no-op",
    "list_document_chunks": "no-op",
    "list_documents": "adapted",
    "list_entities": "no-op",
    "list_memories": "native",
    "list_mental_models": "no-op",
    "list_operations": "adapted",
    "list_tags": "adapted",
    "list_webhook_deliveries": "no-op",
    "list_webhooks": "no-op",
    "metrics_endpoint_metrics_get": "adapted",
    "recall_memories": "native",
    "recover_consolidation": "no-op",
    "reflect": "adapted",
    "refresh_mental_model": "no-op",
    "regenerate_entity_observations": "no-op",
    "reprocess_by_projection": "no-op",
    "reprocess_document": "no-op",
    "reset_bank_config": "adapted",
    "retain_memories": "native",
    "retry_operation": "adapted",
    "trigger_consolidation": "no-op",
    "update_bank": "adapted",
    "update_bank_config": "adapted",
    "update_bank_disposition": "adapted",
    "update_directive": "no-op",
    "update_document": "adapted",
    "update_mental_model": "no-op",
    "update_webhook": "no-op",
}

# Our verdict per operationId. Reasons are grouped as comment blocks: every entry under a block shares its reason.
# 我们对每个 operationId 的判定。理由按注释块分组：块下的每个条目共享该理由。
CAPABILITY: dict[str, str] = {
    # Mem0 pass-through, upstream-native: HMS implements the same five Mem0 methods over its own dual-plane store,
    # so the read/write semantics carry over unchanged (HMS.get_all / get / history / search / add / delete_all).
    # Mem0 直通，上游即 native：HMS 在自己的双平面存储上实现同样的五个 Mem0 方法，
    # 因此读写语义原样成立（HMS.get_all / get / history / search / add / delete_all）。
    "list_memories": "native",
    "get_memory": "native",
    "get_observation_history": "native",
    "recall_memories": "native",
    "retain_memories": "native",
    "clear_bank_memories": "native",
    # Entities: upstream no-ops. We keep a first-class `entities` table, an `atom_entity` link table and one
    # consolidation-minted entity atom per entity, so canonical_name / mention_count / first_seen / last_seen /
    # observations[] are all read straight off the engine rather than reconstructed.
    # 实体：上游是 no-op。我们有一等公民的 `entities` 表、`atom_entity` 链接表，以及 consolidation 为每个实体
    # 生成的 entity 原子，因此 canonical_name / mention_count / first_seen / last_seen / observations[] 全部
    # 直接从引擎读出，而非重构出来。
    "list_entities": "native",
    "get_entity": "native",
    # Entity co-occurrence graph: upstream no-op. Two entities that appear on the same atom co-occur, and the
    # atom's own witness anchors carry the evidence, so the weighted edge is a plain query over atom_entity.
    # 实体共现图：上游是 no-op。出现在同一原子上的两个实体即为共现，该原子自身的见证锚点承载证据，
    # 因此带权边只是对 atom_entity 的一次普通查询。
    "get_entity_graph": "native",
    # Memory graph: upstream no-op. graph_nodes / graph_edges are one of our two planes, and every edge carries
    # a non-empty witness anchor set - the property mem0 has no place to store.
    # 记忆图：上游是 no-op。graph_nodes / graph_edges 就是我们两个平面之一，且每条边都带非空的见证锚点集合
    # —— 这正是 mem0 无处安放的属性。
    "get_graph": "native",
    # Consolidation trigger: upstream no-op. HMS.consolidate runs orient/collect/consolidate/prune synchronously
    # and returns a real per-phase report, so there is nothing to fake and nothing to queue.
    # 凝聚触发：上游是 no-op。HMS.consolidate 同步跑 orient/collect/consolidate/prune 并返回真实的分阶段报告，
    # 因此既无需伪造，也无需排队。
    "trigger_consolidation": "native",
    # Bank statistics: upstream fills the shape from service metadata. Our numbers come from the engine itself -
    # total_observations, pending_consolidation (EPs not yet coarsened) and the fact-type histogram are counted rows.
    # 记忆库统计：上游用服务层元数据填形状。我们的数字来自引擎本身 ——
    # total_observations、pending_consolidation（尚未被粗化的 EP）与事实类型直方图都是真实计数。
    "get_agent_stats": "native",
    # Reflect / extract: upstream runs one search plus one answer-model call. HMS.compile is a deterministic
    # state compiler with a witness closure, HMSCapabilities.reflect_evidence hands the answer model the Detective
    # Report as an `organize`-shaped ledger whose every row cites anchors, and HMS.add(infer=True) is the extraction
    # pipeline itself.
    # Reflect / extract：上游是一次检索加一次应答模型调用。HMS.compile 是带见证闭包的确定性状态编译器，
    # HMSCapabilities.reflect_evidence 把侦探报告作为每行都引用锚点的 `organize` 形状账本交给应答模型，
    # 而 HMS.add(infer=True) 本身就是抽取流水线。
    "reflect": "native",
    "extract_memories": "native",
    # Mental models: upstream no-ops. A "stored reflect response" is our skill atom - an InterfaceEP constraint
    # invariant distilled by promote_skills - so the fields exist but under our names; history walks the lineage.
    # 心智模型：上游是 no-op。"已存的 reflect 响应"就是我们的 skill 原子 —— 由 promote_skills 从 InterfaceEP
    # 的约束不变量蒸馏而来 —— 字段都在，只是换了我们的名字；history 走谱系。
    "list_mental_models": "adapted",
    "get_mental_model": "adapted",
    "get_mental_model_history": "adapted",
    # Directives: upstream no-ops. Ours are constraint atoms (hms.obs_type == 'constraint'); is_active comes from
    # the validity window and priority from confidence, so the two integers are derived rather than user-written.
    # 指令：上游是 no-op。我们的指令是 constraint 原子（hms.obs_type == 'constraint'）；is_active 来自有效期，
    # priority 来自 confidence，因此这两个整数是派生出来的，而非用户手写的。
    "list_directives": "adapted",
    "get_directive": "adapted",
    # Observation clearing: upstream no-ops. We archive instead of deleting - HMS never physically removes a row -
    # so deleted_count reports how many consolidation-produced atoms were archived with a witness anchor.
    # 清除 observation：上游是 no-op。我们归档而非删除 —— HMS 从不物理删除行 ——
    # 因此 deleted_count 报告的是有多少 consolidation 产物在带见证锚点的前提下被归档。
    "clear_observations": "adapted",
    "clear_memory_observations": "adapted",
    # Consolidation recovery: upstream no-op. We have no permanent failure state, so recovery degenerates to
    # re-running the phases over whatever is still pending; retried_count is that pending count.
    # 凝聚恢复：上游是 no-op。我们没有永久失败态，因此"恢复"退化为对仍待处理的部分重跑阶段；
    # retried_count 就是这个待处理数。
    "recover_consolidation": "adapted",
    # Tags and time series: the engine holds topic tags on atoms and every atom carries three instants, so both
    # answers are axis queries; the response shape is upstream's, not ours.
    # 标签与时间序列：引擎在原子上保存主题标签，且每个原子带三个时刻，因此两者都是轴查询；
    # 响应形状是上游的，不是我们的。
    "list_tags": "adapted",
    "get_memories_timeseries": "adapted",
    # Authored mental models and authored directives: the engine derives both from evidence and refuses to accept
    # a hand-written one, because an atom without a witness anchor cannot exist. Authoring belongs to the service.
    # 手写心智模型与手写指令：引擎从证据派生二者，且拒绝接受手写内容，因为没有见证锚点的原子不可能存在。
    # 手写属于服务层。
    "create_mental_model": "unsupported",
    "update_mental_model": "unsupported",
    "delete_mental_model": "unsupported",
    "refresh_mental_model": "unsupported",
    "create_directive": "unsupported",
    "update_directive": "unsupported",
    "delete_directive": "unsupported",
    # Per-entity regeneration: consolidation regenerates every entity atom of a scope in one pass; there is no
    # engine operation scoped to a single entity, and inventing one would not match the phase report.
    # 单实体重生成：consolidation 一次性重生成某范围内的全部 entity 原子；引擎没有作用于单个实体的操作，
    # 硬造一个也无法与阶段报告对上。
    "regenerate_entity_observations": "unsupported",
    # Webhooks: delivery, retry and signature state belong to the service layer, not the engine.
    # Webhook：投递、重试与签名状态属于服务层，不在引擎范围。
    "create_webhook": "unsupported",
    "list_webhooks": "unsupported",
    "update_webhook": "unsupported",
    "delete_webhook": "unsupported",
    "list_webhook_deliveries": "unsupported",
    # Bank templates: import/export of a bank blueprint is a service-layer packaging format; our transport format
    # is the StatePackage, which carries a witness closure and is deliberately not the same artifact.
    # 记忆库模板：记忆库蓝图的导入/导出是服务层的打包格式；我们的传输格式是 StatePackage，
    # 它携带见证闭包，刻意不是同一种产物。
    "import_bank_template": "unsupported",
    "export_bank_template": "unsupported",
    "get_bank_template_schema": "unsupported",
    # Documents and chunks: ingestion of files, chunking and re-chunking are service-layer infrastructure; the
    # engine's unit is the anchored SourceEvent, and it has no document identity to hand back.
    # 文档与分块：文件摄取、分块与重分块属于服务层设施；引擎的单位是落锚的 SourceEvent，
    # 它没有可以交还的文档身份。
    "list_documents": "unsupported",
    "get_document": "unsupported",
    "update_document": "unsupported",
    "delete_document": "unsupported",
    "reprocess_document": "unsupported",
    "reprocess_by_projection": "unsupported",
    "list_document_chunks": "unsupported",
    "get_chunk": "unsupported",
    "file_retain": "unsupported",
    # Async operations: our consolidate is synchronous by contract decision, so there is no queue to list, poll,
    # cancel or retry; the operation id we hand back is the scheduler anchor that witnessed the finished run.
    # 异步操作：按契约决定我们的 consolidate 是同步的，因此没有队列可列出、轮询、取消或重试；
    # 我们交还的操作 id 是见证该次已完成运行的调度器锚点。
    "list_operations": "unsupported",
    "get_operation_status": "unsupported",
    "cancel_operation": "unsupported",
    "retry_operation": "unsupported",
    # Audit logs: the engine keeps history rows and anchors, but an audit log is a tenant-scoped service artifact
    # with its own retention and access rules; exposing history as audit would misstate both.
    # 审计日志：引擎保有历史行与锚点，但审计日志是带自身留存与访问规则的租户级服务产物；
    # 把 history 当作审计暴露会同时误述两者。
    "list_audit_logs": "unsupported",
    "audit_log_stats": "unsupported",
    # Bank lifecycle, profile and configuration: a bank is a Scope to us, and a Scope is not an object with its
    # own row, disposition text or config; creating and deleting banks is service-layer tenancy.
    # 记忆库生命周期、画像与配置：记忆库对我们而言就是一个 Scope，而 Scope 不是拥有自身行、
    # 处置文本或配置的对象；创建与删除记忆库属于服务层的租户管理。
    "list_banks": "unsupported",
    "create_or_update_bank": "unsupported",
    "update_bank": "unsupported",
    "delete_bank": "unsupported",
    "get_bank_profile": "unsupported",
    "update_bank_disposition": "unsupported",
    "add_bank_background": "unsupported",
    "get_bank_config": "unsupported",
    "update_bank_config": "unsupported",
    "reset_bank_config": "unsupported",
    # Health, version and metrics: process-level endpoints of the serving shell; a library has no port to report.
    # 健康、版本与指标：服务壳的进程级端点；库没有端口可以汇报。
    "health_endpoint_health_get": "unsupported",
    "get_version": "unsupported",
    "metrics_endpoint_metrics_get": "unsupported",
}

# The HMS entry point behind every operationId we do not mark "unsupported"; the audit trail for the table above.
# 每个我们未标为 "unsupported" 的 operationId 背后的 HMS 入口；上表的审计线索。
SURFACE: dict[str, str] = {
    "list_memories": "HMS.get_all",
    "get_memory": "HMS.get",
    "get_observation_history": "HMS.history",
    "recall_memories": "HMS.search",
    "retain_memories": "HMS.add",
    "clear_bank_memories": "HMS.delete_all",
    "reflect": "HMS.compile + HMSCapabilities.reflect_evidence",
    "extract_memories": "HMS.add(infer=True)",
    "list_tags": "HMS.axis_query(project_axis='topic')",
    "get_memories_timeseries": "HMS.axis_query(project_axis='time')",
    "list_entities": "HMSCapabilities.list_entities",
    "get_entity": "HMSCapabilities.get_entity",
    "get_entity_graph": "HMSCapabilities.get_entity_graph",
    "get_graph": "HMSCapabilities.get_graph",
    "list_mental_models": "HMSCapabilities.list_mental_models",
    "get_mental_model": "HMSCapabilities.get_mental_model",
    "get_mental_model_history": "HMSCapabilities.get_mental_model_history",
    "list_directives": "HMSCapabilities.list_directives",
    "get_directive": "HMSCapabilities.get_directive",
    "trigger_consolidation": "HMSCapabilities.trigger_consolidation",
    "recover_consolidation": "HMSCapabilities.recover_consolidation",
    "clear_observations": "HMSCapabilities.clear_observations",
    "clear_memory_observations": "HMSCapabilities.clear_memory_observations",
    "get_agent_stats": "HMSCapabilities.bank_stats",
}


def compare_with_upstream() -> dict:
    """Statistics of CAPABILITY against UPSTREAM: how many of their no-ops this engine lifts, and what it drops.
    以 CAPABILITY 对比 UPSTREAM 的统计：本引擎点亮了他们多少个 no-op，又放弃了哪些。

    Input / 输入：
        none.
        无。

    Output / 输出：
        {'total', 'upstream_counts', 'our_counts', 'lifted', 'lifted_count', 'still_unsupported', 'dropped', 'missing'}.
        {'total', 'upstream_counts', 'our_counts', 'lifted', 'lifted_count', 'still_unsupported', 'dropped', 'missing'}。

    State effect / 状态影响：
        None; the function only reads the module-level tables.
        无；本函数只读取模块级的表。
    """
    ours = {name: CAPABILITY.get(name, "unsupported") for name in UPSTREAM}
    lifted: dict[str, list[str]] = {"native": [], "adapted": []}
    still_unsupported: list[str] = []
    dropped: list[str] = []
    for name in sorted(UPSTREAM):
        upstream, mine = UPSTREAM[name], ours[name]
        if upstream == "no-op":
            if mine == "unsupported":
                still_unsupported.append(name)
            else:
                lifted[mine].append(name)
        elif mine == "unsupported":
            dropped.append(name)
    upstream_counts = {verdict: sum(1 for v in UPSTREAM.values() if v == verdict) for verdict in ("native", "adapted")}
    upstream_counts["no-op"] = sum(1 for v in UPSTREAM.values() if v == "no-op")
    our_counts = {verdict: sum(1 for v in ours.values() if v == verdict) for verdict in VERDICTS}
    return {
        "total": len(UPSTREAM),
        "upstream_counts": upstream_counts,
        "our_counts": our_counts,
        "lifted": lifted,
        "lifted_count": len(lifted["native"]) + len(lifted["adapted"]),
        "still_unsupported": still_unsupported,
        "dropped": dropped,
        "missing": sorted(set(UPSTREAM) - set(CAPABILITY)),
    }


__all__ = ["CAPABILITY", "SURFACE", "UPSTREAM", "VERDICTS", "compare_with_upstream"]
