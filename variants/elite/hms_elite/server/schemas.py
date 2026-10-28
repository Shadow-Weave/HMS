"""Request and response schemas of the HTTP service face (pydantic models mirroring the HMS facade arguments).
HTTP 服务面的请求与响应模式（镜像 HMS 门面参数的 pydantic 模型）。
"""

# Purpose (EN): Declare every JSON body the server accepts and the fixed-shape envelopes it returns.
# 目的（中）：声明服务端接受的每种 JSON 请求体，以及它返回的固定形状信封。

# Input (EN): JSON bodies decoded by FastAPI; scope ids and Mem0 filter dicts.
# 输入（中）：由 FastAPI 解码的 JSON 请求体；范围 id 与 Mem0 过滤字典。

# Output (EN): Validated request models (unknown keys rejected) and response models used for OpenAPI + validation.
# 输出（中）：校验后的请求模型（拒绝未知键）与用于 OpenAPI 及校验的响应模型。

# State effect (EN): None; pure data classes.
# 状态影响（中）：无；纯数据类。

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# The three Mem0 scope ids, in the order the facade and the CLI use them.
# 三个 Mem0 范围 id，顺序与门面和命令行一致。
SCOPE_KEYS = ("user_id", "agent_id", "run_id")


def merge_scope(
    filters: dict[str, Any] | None,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Merge explicit scope ids over a Mem0 filter dict; explicit non-empty ids win over keys already in filters.
    将显式范围 id 合并到 Mem0 过滤字典之上；非空的显式 id 优先于过滤字典中已有的键。

    Input / 输入：
        filters: Mem0 filter dict or None.
        filters：Mem0 过滤字典或 None。

        user_id: Explicit user id (None/empty → not merged).
        user_id：显式用户 id（None/空 → 不合并）。

        agent_id: Explicit agent id.
        agent_id：显式代理 id。

        run_id: Explicit run id.
        run_id：显式运行 id。

    Output / 输出：
        A new dict; scope validation itself is left to the facade (VALIDATION_001 when no scope key is present).
        一个新字典；范围校验本身交给门面（没有范围键时 VALIDATION_001）。

    State effect / 状态影响：
        None.
        无。
    """
    merged: dict[str, Any] = dict(filters or {})
    for key, value in zip(SCOPE_KEYS, (user_id, agent_id, run_id)):
        if value:
            merged[key] = value
    return merged


class StrictBody(BaseModel):
    """Base request body: unknown keys are rejected so typos surface as 422 instead of being ignored.
    请求体基类：拒绝未知键，使拼写错误以 422 暴露而不是被静默忽略。
    """

    model_config = ConfigDict(extra="forbid")


class ScopedBody(StrictBody):
    """Request body carrying the three Mem0 scope ids and an optional Mem0 filter dict.
    携带三个 Mem0 范围 id 与可选 Mem0 过滤字典的请求体。
    """

    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    filters: dict[str, Any] | None = None

    def scope_filters(self) -> dict[str, Any]:
        """The Mem0 filter dict handed to the facade: `filters` with the explicit scope ids merged over it.
        交给门面的 Mem0 过滤字典：`filters` 与合并于其上的显式范围 id。

        Input / 输入：
            none.
            无。

        Output / 输出：
            A new dict (see merge_scope).
            一个新字典（见 merge_scope）。

        State effect / 状态影响：
            None.
            无。
        """
        return merge_scope(self.filters, user_id=self.user_id, agent_id=self.agent_id, run_id=self.run_id)


class AddMemoriesRequest(StrictBody):
    """POST /memories body (Mem0 add()).
    POST /memories 的请求体（Mem0 的 add()）。
    """

    messages: str | list[dict[str, Any]] | dict[str, Any]
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] | None = None
    timestamp: str | None = None
    expiration_date: str | None = None
    infer: bool = True
    memory_type: str | None = None
    prompt: str | None = None


class SearchRequest(ScopedBody):
    """POST /search body (Mem0 search()).
    POST /search 的请求体（Mem0 的 search()）。
    """

    query: str
    top_k: int = 20
    threshold: float = 0.1
    rerank: bool = False
    explain: bool = False
    reference_date: str | None = None
    show_expired: bool = False


class UpdateMemoryRequest(StrictBody):
    """PUT /memories/{id} body (Mem0 update()); an absent expiration_date is untouched, null clears it.
    PUT /memories/{id} 的请求体（Mem0 的 update()）；不传 expiration_date 表示不变，传 null 表示清除。
    """

    text: str | None = None
    metadata: dict[str, Any] | None = None
    expiration_date: str | None = None


class EventsRequest(StrictBody):
    """POST /events body (HMS add_events()).
    POST /events 的请求体（HMS 的 add_events()）。
    """

    events: list[dict[str, Any]]
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    infer: bool = True


class CompileRequest(ScopedBody):
    """POST /compile body (HMS compile()).
    POST /compile 的请求体（HMS 的 compile()）。
    """

    query: str
    budget_tokens: int | None = None
    mode: str | None = None
    as_of: str | None = None
    require_witness: bool | None = None


class FeedbackRequest(StrictBody):
    """POST /feedback body (HMS feedback()).
    POST /feedback 的请求体（HMS 的 feedback()）。
    """

    target_id: str | None = None
    feedback: str
    reason: str | None = None
    correction: str | dict[str, Any] | None = None
    actor_id: str | None = None


class CheckpointRequest(ScopedBody):
    """POST /checkpoint body (HMS checkpoint()).
    POST /checkpoint 的请求体（HMS 的 checkpoint()）。
    """

    label: str | None = None


class RollbackRequest(StrictBody):
    """POST /rollback body (HMS rollback()).
    POST /rollback 的请求体（HMS 的 rollback()）。
    """

    checkpoint_id: str
    reason: str | None = None


class HandoffRequest(ScopedBody):
    """POST /handoff body (HMS handoff()).
    POST /handoff 的请求体（HMS 的 handoff()）。
    """

    goal: str | None = None
    budget_tokens: int | None = None
    checkpoint: bool = True
    include_edges: bool = True


class ConsolidateRequest(ScopedBody):
    """POST /consolidate body (HMS consolidate()); an empty scope means every scope in the store.
    POST /consolidate 的请求体（HMS 的 consolidate()）；范围为空表示存储中的每个范围。
    """

    phases: list[str] | None = None


class UsageStage(BaseModel):
    """One telemetry stage of a request, mirrored from telemetry.StageRecord.
    请求的一个遥测阶段，镜像自 telemetry.StageRecord。
    """

    operation: str             # stage name, e.g. 'llm.generate' / 阶段名，如 'llm.generate'
    duration_ms: float         # wall time of this stage / 本阶段墙钟耗时
    input_tokens: int | None   # None when no tokens were reported / 未报告 token 时为 None
    output_tokens: int | None  # None for embedding stages / 嵌入阶段为 None
    total_tokens: int | None   # input + output as reported / 提供者报告的 input + output
    provider_calls: int        # provider calls folded in (0 for a plain stage) / 折入的调用数（普通 stage 为 0）
    estimated: bool            # true when the tokens are estimates / token 为估算值时为真
    error: str | None          # failure text when this stage raised / 本阶段抛错时的文本


class TokenUsage(BaseModel):
    """Real token consumption of one request, mirrored from telemetry.RequestTrace.to_dict().
    单次请求的真实 token 消耗，镜像自 telemetry.RequestTrace.to_dict()。
    """

    request_id: str            # correlating id, echoed as X-Request-ID / 关联 id，回写为 X-Request-ID
    llm_input_tokens: int      # LLM prompt tokens / LLM 提示 token
    llm_output_tokens: int     # LLM completion tokens / LLM 补全 token
    embedding_tokens: int      # embedder input tokens, billed per input / 嵌入器输入 token（只按输入计费）
    total_tokens: int          # the three above, summed / 以上三项之和
    estimated: bool            # true when ANY stage was estimated / 任一阶段为估算值时为真
    stages: list[UsageStage]   # per-stage detail, for cost attribution / 逐阶段明细，便于归因成本


class ResultsResponse(BaseModel):
    """Mem0 `{'results': [...]}` envelope (rows keep every key the facade returns).
    Mem0 的 `{'results': [...]}` 信封（行保留门面返回的每个键）。
    """

    results: list[dict[str, Any]]


class UsageResultsResponse(ResultsResponse):
    """ResultsResponse plus this request's token usage (the token-consuming routes that return `results`).
    在 ResultsResponse 之上附带本请求的 token 用量（返回 `results` 的耗 token 路由）。
    """

    usage: TokenUsage


class MessageResponse(BaseModel):
    """Mem0 `{'message': ...}` envelope.
    Mem0 的 `{'message': ...}` 信封。
    """

    message: str


class UsageMessageResponse(MessageResponse):
    """MessageResponse plus this request's token usage (PUT /memories/{id}, which re-embeds the new text).
    在 MessageResponse 之上附带本请求的 token 用量（PUT /memories/{id}，会重新嵌入新文本）。
    """

    usage: TokenUsage


class HealthResponse(BaseModel):
    """GET /health body: liveness, package version and the configured provider names (llm / embedder / vector_store).
    GET /health 的响应体：存活状态、包版本与配置的提供者名（llm / embedder / vector_store）。
    """

    status: str
    version: str
    providers: dict[str, str]


class UsageResponse(BaseModel):
    """GET /usage body: telemetry.Usage.to_dict() verbatim (merged llm + embedder meters of the served HMS).
    GET /usage 的响应体：与 telemetry.Usage.to_dict() 逐字一致（所服务 HMS 的 llm + embedder 计量器合并值）。
    """

    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated: bool
    embedding_calls: int
    embedded_texts: int
    duration_ms: float


class CompileResponse(BaseModel):
    """POST /compile body: the ContextIR as JSON, the rendered Work State Card, plus this request's token usage.
    POST /compile 的响应体：JSON 形式的 ContextIR、渲染后的工作状态卡，以及本请求的 token 用量。
    """

    context: dict[str, Any]
    card: str
    usage: TokenUsage


class CheckpointResponse(BaseModel):
    """POST /checkpoint body.
    POST /checkpoint 的响应体。
    """

    checkpoint_id: str


class ErrorResponse(BaseModel):
    """Error body: HMSError.to_dict() verbatim (documented on every route for OpenAPI).
    错误响应体：与 HMSError.to_dict() 逐字一致（在每条路由的 OpenAPI 中记录）。
    """

    error_type: str
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    suggestion: str | None = None
    debug_info: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "SCOPE_KEYS",
    "AddMemoriesRequest",
    "CheckpointRequest",
    "CheckpointResponse",
    "CompileRequest",
    "CompileResponse",
    "ConsolidateRequest",
    "ErrorResponse",
    "EventsRequest",
    "FeedbackRequest",
    "HandoffRequest",
    "HealthResponse",
    "MessageResponse",
    "ResultsResponse",
    "RollbackRequest",
    "ScopedBody",
    "SearchRequest",
    "StrictBody",
    "TokenUsage",
    "UpdateMemoryRequest",
    "UsageMessageResponse",
    "UsageResponse",
    "UsageResultsResponse",
    "UsageStage",
    "merge_scope",
]
