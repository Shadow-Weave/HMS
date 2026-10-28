"""FastAPI application over one HMS facade: the Mem0 self-hosted endpoint shapes plus the HMS-native surface.
基于单个 HMS 门面的 FastAPI 应用：Mem0 自托管服务的端点形状，加上 HMS 原生接口。
"""

# Purpose (EN): The `server` extra (CONTRACT §0.5 #16 reserved it; this module fills it): create_app() builds the
#               app, main() runs it under uvicorn as the `hms-server` console script.
# 目的（中）：`server` extra（契约 §0.5 #16 预留了它；本模块将其补齐）：create_app() 构建应用，
#            main() 作为 `hms-server` 控制台脚本在 uvicorn 下运行它。

# Input (EN): An HMS instance (or an HMS config dict), JSON request bodies / query strings, HMS_SERVER_TOKEN,
#             an optional X-Request-ID request header.
# 输入（中）：一个 HMS 实例（或 HMS 配置字典）、JSON 请求体 / 查询串、环境变量 HMS_SERVER_TOKEN、
#            可选的 X-Request-ID 请求头。

# Output (EN): JSON responses carrying the facade's envelopes verbatim, and the token-consuming routes adding a
#              'usage' entry (RequestTrace.to_dict() shape) beside them; errors as HMSError.to_dict() with
#              ValidationError → 400, NOT_FOUND_* → 404, AuthError → 401, other HMSError → 500; every response
#              carries X-Request-ID; GET /usage (Usage JSON) and GET /metrics (Prometheus text) expose telemetry.
# 输出（中）：逐字携带门面信封的 JSON 响应，且消耗 token 的路由在其旁追加 'usage' 条目
#            （RequestTrace.to_dict() 形状）；错误为 HMSError.to_dict()，其中 ValidationError → 400、
#            NOT_FOUND_* → 404、AuthError → 401、其余 HMSError → 500；每个响应携带 X-Request-ID；
#            GET /usage（Usage JSON）与 GET /metrics（Prometheus 文本）暴露遥测数据。

# State effect (EN): Every route delegates to the facade (its transactions apply); importing this module has no
#                    side effects; create_app() without an HMS opens the default store ($HMS_DIR/hms.db).
# 状态影响（中）：每条路由都委托给门面（沿用其事务）；导入本模块没有副作用；
#                未传入 HMS 时 create_app() 打开默认存储（$HMS_DIR/hms.db）。

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import pathlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from typing import Any

from hms_elite import __version__
from hms_elite.exceptions import DependencyError, HMSError, ValidationError, unwrap_hms_error
from hms_elite.memory.main import HMS
from hms_elite.memory.utils import _UNSET
from hms_elite.server.deps import (
    METRICS_CONTENT_TYPE,
    REQUEST_ID_HEADER,
    REQUEST_ID_MAX_LEN,
    AuthError,
    RequestMetrics,
    get_hms,
    get_metrics,
    request_usage,
    require_bearer,
    resolve_token,
)
from hms_elite.server.schemas import (
    AddMemoriesRequest,
    CheckpointRequest,
    CheckpointResponse,
    CompileRequest,
    CompileResponse,
    ConsolidateRequest,
    ErrorResponse,
    EventsRequest,
    FeedbackRequest,
    HandoffRequest,
    HealthResponse,
    MessageResponse,
    ResultsResponse,
    RollbackRequest,
    SearchRequest,
    UpdateMemoryRequest,
    UsageMessageResponse,
    UsageResponse,
    UsageResultsResponse,
    merge_scope,
)
from hms_elite.telemetry import trace

logger = logging.getLogger(__name__)

# Mem0-style import guard: the check runs at module import; the error is raised when create_app() runs so that the
# module itself stays importable (test_bilingual_docstrings imports every module).
# Mem0 风格的导入守卫：检查在模块导入时执行，错误推迟到 create_app() 运行时抛出，
# 从而让模块本身保持可导入（test_bilingual_docstrings 会导入每一个模块）。
_IMPORT_ERROR: ImportError | None = None
try:
    from fastapi import APIRouter, Depends, FastAPI, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse, PlainTextResponse
except ImportError as _exc:  # pragma: no cover - depends on the environment
    APIRouter = Depends = FastAPI = JSONResponse = PlainTextResponse = jsonable_encoder = None  # type: ignore
    Request = Any  # type: ignore[misc,assignment]
    _IMPORT_ERROR = _exc

# ASGI type aliases (kept local so the module needs no starlette import to be readable).
# ASGI 类型别名（保持本地定义，使模块无需导入 starlette 也可读）。
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
# Operation label of a request that matched no route (404 from the router).
# 未匹配任何路由（路由器返回 404）的请求的操作标签。
UNMATCHED_OPERATION = "unmatched"

RESET_MESSAGE = "All memories reset"
# The OpenAPI error documentation attached to every authenticated route.
# 附加到每条受鉴权路由上的 OpenAPI 错误文档。
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "ValidationError (VALIDATION_*)"},
    401: {"model": ErrorResponse, "description": "AuthError (AUTH_001)"},
    404: {"model": ErrorResponse, "description": "Not found (NOT_FOUND_*)"},
    500: {"model": ErrorResponse, "description": "Other HMSError"},
}


# ------------------------------------------------------------------ error mapping
# ------------------------------------------------------------------ 错误映射
def error_status(error: HMSError) -> int:
    """HTTP status of an HMSError: AuthError 401, NOT_FOUND_* 404, ValidationError 400, everything else 500.
    HMSError 对应的 HTTP 状态码：AuthError 401、NOT_FOUND_* 404、ValidationError 400、其余 500。

    Input / 输入：
        error: The library error.
        error：库错误。

    Output / 输出：
        The status code.
        状态码。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(error, AuthError):
        return 401
    if error.error_code.startswith("NOT_FOUND"):
        return 404
    if isinstance(error, ValidationError):
        return 400
    return 500


INTERNAL_MESSAGE = "internal error"
INTERNAL_CODE = "INTERNAL_000"


def error_response(exc: BaseException, *, request_id: str | None = None) -> JSONResponse:
    """Map any exception the facade raised to a JSON response whose body is HMSError.to_dict(); a 500 is redacted.
    把门面抛出的任意异常映射为响应体等于 HMSError.to_dict() 的 JSON 响应；500 会被脱敏。

    Input / 输入：
        exc: An HMSError (or a pydantic error wrapping one), a bare ValueError (→ VALIDATION_000), or any other exception (→ INTERNAL_000).
        exc：HMSError（或包裹着它的 pydantic 错误）、裸 ValueError（→ VALIDATION_000），或任何其它异常（→ INTERNAL_000）。

        request_id: The X-Request-ID of the request, logged with every 500.
        request_id：请求的 X-Request-ID，随每个 500 一起记录日志。

    Output / 输出：
        JSONResponse with the status of error_status() and the to_dict() body. A 500 body keeps only `error_type` and
        `error_code`: `message` becomes INTERNAL_MESSAGE, `details` / `suggestion` / `debug_info` are emptied (a
        StoreError carries the database path and SQL fragments; they are logged server-side, never sent).
        状态码为 error_status()、响应体为 to_dict() 的 JSONResponse。500 的响应体只保留 `error_type` 与
        `error_code`：`message` 变为 INTERNAL_MESSAGE，`details` / `suggestion` / `debug_info` 清空（StoreError 携带
        数据库路径与 SQL 片段；它们只记在服务端日志里，绝不发出）。

    State effect / 状态影响：
        Logs every 500 (full error, exc_info) with the request id.
        以请求 id 记录每个 500（完整错误、exc_info）。
    """
    error = unwrap_hms_error(exc)
    if error is None:
        if isinstance(exc, ValueError):
            error = ValidationError(str(exc), "VALIDATION_000", details={"exception": type(exc).__name__})
        else:
            error = HMSError(INTERNAL_MESSAGE, INTERNAL_CODE, details={"exception": type(exc).__name__})
    status = error_status(error)
    body = error.to_dict()
    if status == 500:
        logger.error("request %s failed: %r %s", request_id, error, body.get("details"), exc_info=exc)
        body = {
            **body,
            "message": INTERNAL_MESSAGE,
            "details": {},
            "suggestion": None,
            "debug_info": {},
        }
    return JSONResponse(status_code=status, content=jsonable_encoder(body))


async def _handle_error(request: Request, exc: Exception) -> JSONResponse:
    """Starlette exception handler bound to HMSError and ValueError (RequestTelemetry catches everything else).
    绑定到 HMSError 与 ValueError 的 Starlette 异常处理器（其余一切由 RequestTelemetry 捕获）。
    """
    return error_response(exc, request_id=_request_id(request.scope))


def _csv(value: str | None) -> list[str] | None:
    """Comma-separated query parameter → list (None when empty).
    逗号分隔的查询参数 → 列表（为空时 None）。
    """
    items = [part.strip() for part in (value or "").split(",") if part.strip()]
    return items or None


# ------------------------------------------------------------------ Mem0-shaped routes
# ------------------------------------------------------------------ Mem0 形状的路由
def get_health(request: Request) -> dict[str, Any]:
    """GET /health: liveness, the package version and the configured provider names (never behind the bearer check).
    GET /health：存活状态、包版本与配置的提供者名（永远不经过 Bearer 鉴权）。

    Input / 输入：
        request: The incoming request (carries the HMS facade whose config names the providers).
        request：传入的请求（携带 HMS 门面，其配置给出提供者名）。

    Output / 输出：
        {'status': 'ok', 'version': __version__, 'providers': {'llm', 'embedder', 'vector_store'}}.
        {'status': 'ok', 'version': __version__, 'providers': {'llm', 'embedder', 'vector_store'}}。

    State effect / 状态影响：
        None.
        无。
    """
    config = get_hms(request).config
    providers = {
        "llm": config.llm.provider,
        "embedder": config.embedder.provider,
        "vector_store": config.vector_store.provider,
    }
    return {"status": "ok", "version": __version__, "providers": providers}


def get_usage(request: Request) -> dict[str, Any]:
    """GET /usage: hms.usage() — the merged llm + embedder meters of the served facade, as JSON.
    GET /usage：hms.usage()——所服务门面的 llm + embedder 计量器合并值，以 JSON 返回。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        Usage.to_dict() (llm_calls, input/output/total_tokens, estimated, embedding_calls, embedded_texts, duration_ms).
        Usage.to_dict()（llm_calls、input/output/total_tokens、estimated、embedding_calls、embedded_texts、duration_ms）。

    State effect / 状态影响：
        None (the meters are snapshotted, not reset).
        无（对计量器做快照，不重置）。
    """
    return get_hms(request).usage().to_dict()


def get_metrics_text(request: Request) -> PlainTextResponse:
    """GET /metrics: the Prometheus text exposition of app.state.metrics (requests seen by this process).
    GET /metrics：app.state.metrics 的 Prometheus 文本暴露格式（本进程处理过的请求）。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        text/plain; version=0.0.4 body: hms_llm_calls_total, hms_llm_tokens_total{direction,estimated},
        hms_embedding_calls_total, hms_request_duration_ms{operation} histogram.
        text/plain; version=0.0.4 响应体：hms_llm_calls_total、hms_llm_tokens_total{direction,estimated}、
        hms_embedding_calls_total、hms_request_duration_ms{operation} 直方图。

    State effect / 状态影响：
        None.
        无。
    """
    return PlainTextResponse(get_metrics(request).render(), media_type=METRICS_CONTENT_TYPE)


def post_memories(body: AddMemoriesRequest, request: Request) -> dict[str, Any]:
    """POST /memories: Mem0 add().
    POST /memories：Mem0 的 add()。

    Input / 输入：
        body: AddMemoriesRequest (messages, scope ids and the other add() keyword arguments).
        body：AddMemoriesRequest（messages、范围 id 与 add() 的其余关键字参数）。

        request: The incoming request (carries the HMS facade).
        request：传入的请求（携带 HMS 门面）。

    Output / 输出：
        {'results': [row + 'event' ...]} verbatim from HMS.add, plus 'usage' (the TokenUsage this add spent).
        与 HMS.add 逐字一致的 {'results': [行 + 'event' ...]}，另加 'usage'（本次 add 消耗的 TokenUsage）。

    State effect / 状态影响：
        Writes through HMS.add (Txn A + Txn B).
        通过 HMS.add 写入（事务 A + 事务 B）。
    """
    result = get_hms(request).add(
        body.messages,
        user_id=body.user_id,
        agent_id=body.agent_id,
        run_id=body.run_id,
        metadata=body.metadata,
        timestamp=body.timestamp,
        expiration_date=body.expiration_date,
        infer=body.infer,
        memory_type=body.memory_type,
        prompt=body.prompt,
    )
    # Written last so the usage reflects every provider call this add made (extraction LLM + embeddings).
    # 最后写入，使 usage 覆盖本次 add 的全部提供者调用（抽取 LLM + 嵌入）。
    return {**result, "usage": request_usage()}


def get_memories(
    request: Request,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    top_k: int = 20,
    show_expired: bool = False,
) -> dict[str, Any]:
    """GET /memories: Mem0 get_all() over the query-string scope.
    GET /memories：基于查询串范围的 Mem0 get_all()。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

        user_id: Scope user id (at least one scope id is required, VALIDATION_001 → 400).
        user_id：范围用户 id（至少一个范围 id，否则 VALIDATION_001 → 400）。

        agent_id: Scope agent id.
        agent_id：范围代理 id。

        run_id: Scope run id.
        run_id：范围运行 id。

        top_k: Result cap.
        top_k：结果上限。

        show_expired: Include archived / expired versions.
        show_expired：包含已归档 / 已过期版本。

    Output / 输出：
        {'results': [row...]}.
        {'results': [行...]}。

    State effect / 状态影响：
        None.
        无。
    """
    filters = merge_scope(None, user_id=user_id, agent_id=agent_id, run_id=run_id)
    return get_hms(request).get_all(filters=filters, top_k=top_k, show_expired=show_expired)


def get_memory(memory_id: str, request: Request) -> dict[str, Any]:
    """GET /memories/{memory_id}: Mem0 get(); a missing lineage is 404 instead of a JSON null.
    GET /memories/{memory_id}：Mem0 的 get()；谱系不存在时返回 404 而不是 JSON null。

    Input / 输入：
        memory_id: Mem0 memory id (lineage id).
        memory_id：Mem0 记忆 id（谱系 id）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        The head row; NOT_FOUND_001 (404) when unknown, deleted or fully archived.
        头版本行；未知、已删除或全部归档时 NOT_FOUND_001（404）。

    State effect / 状态影响：
        None.
        无。
    """
    row = get_hms(request).get(memory_id)
    if row is None:
        raise ValidationError(
            f"Memory with id {memory_id} not found", "NOT_FOUND_001", details={"memory_id": memory_id}
        )
    return row


def post_search(body: SearchRequest, request: Request) -> dict[str, Any]:
    """POST /search: Mem0 search().
    POST /search：Mem0 的 search()。

    Input / 输入：
        body: SearchRequest (query, scope ids / filters and the other search() keyword arguments).
        body：SearchRequest（query、范围 id / filters 与 search() 的其余关键字参数）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'results': [row + 'score' (+ 'score_details')], 'usage': the query embedding, plus the reranker LLM when
        rerank=true and an LLM reranker is configured}.
        {'results': [行 + 'score'（+ 'score_details'）], 'usage'：查询嵌入，以及 rerank=true 且配置了
        LLM 重排器时的重排 LLM}。

    State effect / 状态影响：
        None.
        无。
    """
    result = get_hms(request).search(
        body.query,
        top_k=body.top_k,
        filters=body.scope_filters(),
        threshold=body.threshold,
        rerank=body.rerank,
        explain=body.explain,
        reference_date=body.reference_date,
        show_expired=body.show_expired,
    )
    # Query embedding only, unless rerank=true with an LLM reranker configured; usage shows which happened.
    # 通常只有查询嵌入；仅当 rerank=true 且配置了 LLM 重排器时才有 LLM；usage 会把实际情况显示出来。
    return {**result, "usage": request_usage()}


def put_memory(memory_id: str, body: UpdateMemoryRequest, request: Request) -> dict[str, Any]:
    """PUT /memories/{memory_id}: Mem0 update(); expiration_date absent → untouched, null → cleared.
    PUT /memories/{memory_id}：Mem0 的 update()；不传 expiration_date → 不变，null → 清除。

    Input / 输入：
        memory_id: Mem0 memory id (unknown → 404).
        memory_id：Mem0 记忆 id（未知 → 404）。

        body: UpdateMemoryRequest (text, metadata, expiration_date).
        body：UpdateMemoryRequest（text、metadata、expiration_date）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'message': 'Memory updated successfully!', 'usage': the re-embedding a text update costs}.
        {'message': 'Memory updated successfully!', 'usage'：改写文本重新嵌入的花费}。

    State effect / 状态影响：
        Writes through HMS.update (one transaction).
        通过 HMS.update 写入（一个事务）。
    """
    # Only a key present in the JSON body reaches update(); the facade's _UNSET sentinel keeps valid_to untouched.
    # 只有 JSON 请求体中出现的键才会传给 update()；门面的 _UNSET 哨兵让 valid_to 保持不变。
    expiration = body.expiration_date if "expiration_date" in body.model_fields_set else _UNSET
    result = get_hms(request).update(memory_id, text=body.text, metadata=body.metadata, expiration_date=expiration)
    # A text update re-embeds the new version; a metadata- or expiration-only edit spends nothing.
    # 改文本会重新嵌入新版本；只改元数据或有效期则不消耗 token。
    return {**result, "usage": request_usage()}


def get_history(memory_id: str, request: Request) -> list[dict[str, Any]]:
    """GET /memories/{memory_id}/history: Mem0 history() (bare list; unknown id → []).
    GET /memories/{memory_id}/history：Mem0 的 history()（裸列表；未知 id → []）。

    Input / 输入：
        memory_id: Mem0 memory id.
        memory_id：Mem0 记忆 id。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        History rows in seq order.
        按 seq 顺序的历史行。

    State effect / 状态影响：
        None.
        无。
    """
    return get_hms(request).history(memory_id)


def delete_memory(memory_id: str, request: Request) -> dict[str, Any]:
    """DELETE /memories/{memory_id}: Mem0 delete() (archive, never a physical delete).
    DELETE /memories/{memory_id}：Mem0 的 delete()（归档，从不物理删除）。

    Input / 输入：
        memory_id: Mem0 memory id (unknown/deleted → 404).
        memory_id：Mem0 记忆 id（未知/已删除 → 404）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'message': 'Memory deleted successfully!'}.
        {'message': 'Memory deleted successfully!'}。

    State effect / 状态影响：
        Writes through HMS.delete (one transaction).
        通过 HMS.delete 写入（一个事务）。
    """
    return get_hms(request).delete(memory_id)


def delete_memories(
    request: Request, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """DELETE /memories: Mem0 delete_all() over the query-string scope.
    DELETE /memories：基于查询串范围的 Mem0 delete_all()。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

        user_id: Scope user id (at least one scope id is required, VALIDATION_001 → 400).
        user_id：范围用户 id（至少一个范围 id，否则 VALIDATION_001 → 400）。

        agent_id: Scope agent id.
        agent_id：范围代理 id。

        run_id: Scope run id.
        run_id：范围运行 id。

    Output / 输出：
        {'message': 'Memories deleted successfully!'}.
        {'message': 'Memories deleted successfully!'}。

    State effect / 状态影响：
        Writes through HMS.delete_all (one transaction).
        通过 HMS.delete_all 写入（一个事务）。
    """
    return get_hms(request).delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)


def post_reset(request: Request) -> dict[str, Any]:
    """POST /reset: Mem0 reset() (drops every table and clears the index).
    POST /reset：Mem0 的 reset()（删除全部表并清空索引）。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'message': 'All memories reset'} (the Mem0 server literal).
        {'message': 'All memories reset'}（Mem0 服务端的字面值）。

    State effect / 状态影响：
        Destroys all data in the store and the index.
        销毁存储与索引中的全部数据。
    """
    get_hms(request).reset()
    return {"message": RESET_MESSAGE}


# ------------------------------------------------------------------ HMS-native routes
# ------------------------------------------------------------------ HMS 原生路由
def post_events(body: EventsRequest, request: Request) -> dict[str, Any]:
    """POST /events: HMS add_events() (messages and structured tool/file/test/sensor events).
    POST /events：HMS 的 add_events()（消息以及结构化的工具/文件/测试/传感器事件）。

    Input / 输入：
        body: EventsRequest (events, fallback scope ids, infer).
        body：EventsRequest（events、回退范围 id、infer）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'results', 'events', 'eps', 'edges', 'constraints'} verbatim from HMS.add_events, plus 'usage' (nothing
        for structured events, the usual extraction calls for message events with infer=true).
        {'results', 'events', 'eps', 'edges', 'constraints'} 与 HMS.add_events 逐字一致，另加 'usage'
        （结构化事件不消耗；infer=true 的消息事件则执行常规抽取调用）。

    State effect / 状态影响：
        Writes through HMS.add_events.
        通过 HMS.add_events 写入。
    """
    result = get_hms(request).add_events(
        body.events, user_id=body.user_id, agent_id=body.agent_id, run_id=body.run_id, infer=body.infer
    )
    # Structured events take the zero-LLM path; message events with infer=true do the usual extraction calls.
    # 结构化事件走零 LLM 路径；infer=true 的消息事件则执行常规抽取调用。
    return {**result, "usage": request_usage()}


def post_compile(body: CompileRequest, request: Request) -> dict[str, Any]:
    """POST /compile: HMS compile(); returns the ContextIR as JSON and the rendered Work State Card.
    POST /compile：HMS 的 compile()；返回 JSON 形式的 ContextIR 与渲染后的工作状态卡。

    Input / 输入：
        body: CompileRequest (query, scope ids / filters, budget_tokens, mode, as_of, require_witness).
        body：CompileRequest（query、范围 id / filters、budget_tokens、mode、as_of、require_witness）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'context': ContextIR.model_dump(mode='json'), 'card': ContextIR.to_text(), 'usage': the query embedding}.
        {'context': ContextIR.model_dump(mode='json'), 'card': ContextIR.to_text(), 'usage'：查询嵌入}。

    State effect / 状态影响：
        None (read-only).
        无（只读）。
    """
    ir = get_hms(request).compile(
        body.query,
        filters=body.scope_filters(),
        budget_tokens=body.budget_tokens,
        mode=body.mode,
        as_of=body.as_of,
        require_witness=body.require_witness,
    )
    # The compiler itself never calls an LLM; the only spend here is embedding the query text.
    # 编译器自身从不调用 LLM；这里唯一的花费是嵌入查询文本。
    return {"context": ir.model_dump(mode="json"), "card": ir.to_text(), "usage": request_usage()}


def get_why(
    target: str,
    request: Request,
    hops: int = 2,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    edge_types: str | None = None,
    include_invalid: bool = False,
) -> dict[str, Any]:
    """GET /why/{target}: HMS why(); scope ids in the query string enable question mode.
    GET /why/{target}：HMS 的 why()；查询串中的范围 id 启用问题模式。

    Input / 输入：
        target: memory/atom/ep/obs/interface_ep/edge id, or a question when scope ids are given.
        target：memory/atom/ep/obs/interface_ep/edge 的 id；给定范围 id 时也可以是一个问题。

        request: The incoming request.
        request：传入的请求。

        hops: Expansion levels in [0, 6].
        hops：扩展层数，范围 [0, 6]。

        user_id: Scope user id (question mode).
        user_id：范围用户 id（问题模式）。

        agent_id: Scope agent id.
        agent_id：范围代理 id。

        run_id: Scope run id.
        run_id：范围运行 id。

        edge_types: Comma-separated incoming edge types to follow (empty → WHY_EDGE_TYPES).
        edge_types：逗号分隔的、要跟随的入边类型（为空 → WHY_EDGE_TYPES）。

        include_invalid: Also follow invalid edges.
        include_invalid：是否也沿失效边扩展。

    Output / 输出：
        The why() report; NOT_FOUND_001 (404) when nothing resolves.
        why() 报告；无法解析时 NOT_FOUND_001（404）。

    State effect / 状态影响：
        None (read-only).
        无（只读）。
    """
    filters = merge_scope(None, user_id=user_id, agent_id=agent_id, run_id=run_id) or None
    result = get_hms(request).why(
        target, hops=hops, filters=filters, edge_types=_csv(edge_types), include_invalid=include_invalid
    )
    # Only question mode (unresolvable target + scope filters) spends anything: it compiles the question.
    # 只有问题模式（目标无法解析 + 给定范围过滤）才有消耗：它会编译该问题。
    return {**result, "usage": request_usage()}


def post_feedback(body: FeedbackRequest, request: Request) -> dict[str, Any]:
    """POST /feedback: HMS feedback() (one of the eleven state feedback types).
    POST /feedback：HMS 的 feedback()（十一种状态反馈类型之一）。

    Input / 输入：
        body: FeedbackRequest (target_id, feedback, reason, correction, actor_id).
        body：FeedbackRequest（target_id、feedback、reason、correction、actor_id）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'message': 'Feedback applied', 'feedback_id', 'actions': [...], 'usage': the embeddings a correction costs}.
        {'message': 'Feedback applied', 'feedback_id', 'actions': [...], 'usage'：带 correction 时的嵌入花费}。

    State effect / 状态影响：
        Writes through HMS.feedback (one transaction witnessed by the feedback anchor).
        通过 HMS.feedback 写入（一个由反馈锚点见证的事务）。
    """
    result = get_hms(request).feedback(
        body.target_id, body.feedback, reason=body.reason, correction=body.correction, actor_id=body.actor_id
    )
    # Confidence- or topic-only feedback spends nothing; a correction re-embeds every atom it creates or revises.
    # 只改置信度或话题的反馈不消耗；带 correction 时会为每个新建或修订的原子重新嵌入。
    return {**result, "usage": request_usage()}


def post_checkpoint(body: CheckpointRequest, request: Request) -> dict[str, Any]:
    """POST /checkpoint: HMS checkpoint().
    POST /checkpoint：HMS 的 checkpoint()。

    Input / 输入：
        body: CheckpointRequest (scope ids / filters, label).
        body：CheckpointRequest（范围 id / filters、label）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'checkpoint_id': 'ckp_...'}.
        {'checkpoint_id': 'ckp_...'}。

    State effect / 状态影响：
        Writes through HMS.checkpoint (one transaction).
        通过 HMS.checkpoint 写入（一个事务）。
    """
    return {"checkpoint_id": get_hms(request).checkpoint(filters=body.scope_filters(), label=body.label)}


def post_rollback(body: RollbackRequest, request: Request) -> dict[str, Any]:
    """POST /rollback: HMS rollback().
    POST /rollback：HMS 的 rollback()。

    Input / 输入：
        body: RollbackRequest (checkpoint_id, reason).
        body：RollbackRequest（checkpoint_id、reason）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        The rollback report; NOT_FOUND_001 (404) for an unknown checkpoint.
        回滚报告；检查点未知时 NOT_FOUND_001（404）。

    State effect / 状态影响：
        Writes through HMS.rollback (one transaction; content never touched).
        通过 HMS.rollback 写入（一个事务；从不修改内容）。
    """
    return get_hms(request).rollback(body.checkpoint_id, reason=body.reason)


def post_handoff(body: HandoffRequest, request: Request) -> dict[str, Any]:
    """POST /handoff: HMS handoff(); the StatePackage as JSON (StatePackage.model_validate() on the receiver).
    POST /handoff：HMS 的 handoff()；JSON 形式的 StatePackage（接收方用 StatePackage.model_validate()）。

    Input / 输入：
        body: HandoffRequest (scope ids / filters, goal, budget_tokens, checkpoint, include_edges).
        body：HandoffRequest（范围 id / filters、goal、budget_tokens、checkpoint、include_edges）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        StatePackage.model_dump(mode='json').
        StatePackage.model_dump(mode='json')。

    State effect / 状态影响：
        Only the optional checkpoint writes.
        仅可选的检查点会写入。
    """
    package = get_hms(request).handoff(
        filters=body.scope_filters(),
        goal=body.goal,
        budget_tokens=body.budget_tokens,
        checkpoint=body.checkpoint,
        include_edges=body.include_edges,
    )
    # No `usage` here by design: this body IS a StatePackage that the receiver feeds straight into
    # StatePackage.model_validate(), whose extra="forbid" would reject any key we added beside it.
    # 这里刻意不带 `usage`：该响应体本身就是 StatePackage，接收方会直接喂给 StatePackage.model_validate()，
    # 其 extra="forbid" 会拒绝任何并列添加的键。
    return package.model_dump(mode="json")


def get_transitions(
    request: Request,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    since: str | None = None,
    scale: str | None = None,
    fmt: str = "tuples",
    kinds: str | None = None,
) -> Any:
    """GET /transitions: HMS transitions(); fmt=jsonl streams NDJSON text, the other formats return a JSON list.
    GET /transitions：HMS 的 transitions()；fmt=jsonl 返回 NDJSON 文本，其余格式返回 JSON 列表。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

        user_id: Scope user id (at least one scope id is required).
        user_id：范围用户 id（至少一个范围 id）。

        agent_id: Scope agent id.
        agent_id：范围代理 id。

        run_id: Scope run id.
        run_id：范围运行 id。

        since: Inclusive lower bound on time_t1.
        since：time_t1 的下界（含）。

        scale: 'L1_micro' | 'L2_meso' | 'L3_macro' or empty.
        scale：'L1_micro' | 'L2_meso' | 'L3_macro' 或为空。

        fmt: 'tuples' | 'records' | 'jsonl' (VALIDATION_013 → 400 otherwise).
        fmt：'tuples' | 'records' | 'jsonl'（否则 VALIDATION_013 → 400）。

        kinds: Comma-separated source_kind whitelist.
        kinds：逗号分隔的 source_kind 白名单。

    Output / 输出：
        A JSON list (tuples/records) or an application/x-ndjson PlainTextResponse (jsonl).
        JSON 列表（tuples/records）或 application/x-ndjson 的 PlainTextResponse（jsonl）。

    State effect / 状态影响：
        Writes only new transitions rows (INSERT OR IGNORE) through HMS.transitions.
        通过 HMS.transitions 仅写入新的 transitions 行（INSERT OR IGNORE）。
    """
    rows = get_hms(request).transitions(
        filters=merge_scope(None, user_id=user_id, agent_id=agent_id, run_id=run_id),
        since=since,
        scale=scale,
        fmt=fmt,
        kinds=_csv(kinds),
    )
    if isinstance(rows, str):
        return PlainTextResponse(rows, media_type="application/x-ndjson")
    return rows


def post_consolidate(body: ConsolidateRequest, request: Request) -> dict[str, Any]:
    """POST /consolidate: HMS consolidate(); an empty scope runs every scope in the store.
    POST /consolidate：HMS 的 consolidate()；范围为空时运行存储中的每个范围。

    Input / 输入：
        body: ConsolidateRequest (scope ids / filters, phases).
        body：ConsolidateRequest（范围 id / filters、phases）。

        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        {'anchor_id', 'phases': {phase: report}, 'usage': the embeddings, and LLM calls only when collect
        re-ingests pending events or coarsen / causal proposals are configured to 'llm'}.
        {'anchor_id', 'phases': {phase: report}, 'usage'：嵌入，以及仅当 collect 重新抽取待处理事件、
        或 coarsen / 因果提议被配置为 'llm' 时的 LLM 调用}。

    State effect / 状态影响：
        The scheduler phases' writes through HMS.consolidate.
        通过 HMS.consolidate 执行各调度阶段的写入。
    """
    result = get_hms(request).consolidate(filters=body.scope_filters() or None, phases=body.phases)
    # LLM spend only when collect re-ingests pending events, or when coarsen / causal proposals are set to "llm".
    # 仅当 collect 重新抽取待处理事件、或 coarsen / 因果提议被配置为 "llm" 时才有 LLM 消耗。
    return {**result, "usage": request_usage()}


def get_stats(
    request: Request, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """GET /stats: HMS graph_stats(); no scope id means the whole database.
    GET /stats：HMS 的 graph_stats()；没有范围 id 表示整库。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

        user_id: Scope user id.
        user_id：范围用户 id。

        agent_id: Scope agent id.
        agent_id：范围代理 id。

        run_id: Scope run id.
        run_id：范围运行 id。

    Output / 输出：
        The graph statistics dict.
        图统计字典。

    State effect / 状态影响：
        None.
        无。
    """
    filters = merge_scope(None, user_id=user_id, agent_id=agent_id, run_id=run_id) or None
    return get_hms(request).graph_stats(filters=filters)


# (method, path, handler, response_model, operation); order is the Mem0 server order followed by the HMS-native
# routes; `operation` is the facade method name used as the trace operation and the /metrics label.
# （方法、路径、处理函数、响应模型、操作名）；顺序为 Mem0 服务端顺序，随后是 HMS 原生路由；
# `operation` 是门面方法名，用作追踪操作名与 /metrics 标签。
_ROUTES: tuple[tuple[str, str, Callable[..., Any], Any, str], ...] = (
    ("POST", "/memories", post_memories, UsageResultsResponse, "add"),
    ("GET", "/memories", get_memories, ResultsResponse, "get_all"),
    ("GET", "/memories/{memory_id}", get_memory, None, "get"),
    ("POST", "/search", post_search, UsageResultsResponse, "search"),
    ("PUT", "/memories/{memory_id}", put_memory, UsageMessageResponse, "update"),
    ("GET", "/memories/{memory_id}/history", get_history, None, "history"),
    ("DELETE", "/memories/{memory_id}", delete_memory, MessageResponse, "delete"),
    ("DELETE", "/memories", delete_memories, MessageResponse, "delete_all"),
    ("POST", "/reset", post_reset, MessageResponse, "reset"),
    ("POST", "/events", post_events, None, "add_events"),
    ("POST", "/compile", post_compile, CompileResponse, "compile"),
    ("GET", "/why/{target}", get_why, None, "why"),
    ("POST", "/feedback", post_feedback, None, "feedback"),
    ("POST", "/checkpoint", post_checkpoint, CheckpointResponse, "checkpoint"),
    ("POST", "/rollback", post_rollback, None, "rollback"),
    ("POST", "/handoff", post_handoff, None, "handoff"),
    ("GET", "/transitions", get_transitions, None, "transitions"),
    ("POST", "/consolidate", post_consolidate, None, "consolidate"),
    ("GET", "/stats", get_stats, None, "graph_stats"),
    ("GET", "/usage", get_usage, UsageResponse, "usage"),
    ("GET", "/metrics", get_metrics_text, None, "metrics"),
)
# Route name (handler.__name__) → operation label, for the routes above plus /health.
# 路由名（handler.__name__）→ 操作标签，覆盖上表路由与 /health。
_OPERATIONS: dict[str, str] = {handler.__name__: operation for _, _, handler, _, operation in _ROUTES}
_OPERATIONS[get_health.__name__] = "health"


# ------------------------------------------------------------------ request telemetry middleware
# ------------------------------------------------------------------ 请求遥测中间件
def _request_id(scope: Scope) -> str:
    """The client's X-Request-ID when present and sane (non-empty, ≤ REQUEST_ID_MAX_LEN), else a fresh uuid4 hex.
    客户端的 X-Request-ID（非空且 ≤ REQUEST_ID_MAX_LEN 时沿用），否则新生成一个 uuid4 十六进制串。
    """
    wanted = REQUEST_ID_HEADER.lower().encode("latin-1")
    for name, value in scope.get("headers") or ():
        if name.lower() == wanted:
            text = value.decode("latin-1", "replace").strip()
            if 0 < len(text) <= REQUEST_ID_MAX_LEN:
                return text
            break
    return uuid.uuid4().hex


def _operation_of(scope: Scope) -> str:
    """Operation label of the route the request hit, read from scope['endpoint'] after routing; unmatched when none.
    请求命中路由的操作标签，在路由之后从 scope['endpoint'] 读取；没有命中时为 unmatched。
    """
    # Starlette / FastAPI write the matched handler into scope['endpoint'] (stable across routing rewrites, unlike the
    # route objects themselves); a method mismatch (405) still names its route.
    # Starlette / FastAPI 会把命中的处理函数写入 scope['endpoint']（与路由对象本身不同，它跨路由实现改写保持稳定）；
    # 方法不符（405）时依然给出其路由名。
    name = getattr(scope.get("endpoint"), "__name__", None)
    if not name:
        return UNMATCHED_OPERATION
    return _OPERATIONS.get(name, name)


class RequestTelemetry:
    """Pure ASGI middleware: wraps each HTTP request in telemetry.trace(operation, request_id), echoes X-Request-ID.
    纯 ASGI 中间件：把每个 HTTP 请求包进 telemetry.trace(operation, request_id)，并回写 X-Request-ID。
    """

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], metrics: RequestMetrics) -> None:
        """Bind the downstream ASGI app and the registry that receives every finished trace.
        绑定下游 ASGI 应用与接收每条已结束追踪的登记表。
        """
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Open the trace, run the app with a send wrapper adding X-Request-ID, name the operation, observe the trace.
        打开追踪，用添加 X-Request-ID 的 send 包装运行应用，命名操作，再把追踪观测进登记表。
        """
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request_id = _request_id(scope)
        header = (REQUEST_ID_HEADER.lower().encode("latin-1"), request_id.encode("latin-1", "replace"))

        started = False

        async def send_with_id(message: Message) -> None:
            """Append the X-Request-ID header to the response start message.
            向响应起始消息追加 X-Request-ID 头。
            """
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
                headers = [pair for pair in message.get("headers") or [] if pair[0].lower() != header[0]]
                headers.append(header)
                message["headers"] = headers
            await send(message)

        # The trace is set on this task's context before the app runs; Starlette copies the context into the thread
        # that executes a sync route, so provider wrappers inside the facade append their stages to it.
        # 追踪在应用运行前设置于本任务的上下文；Starlette 会把上下文复制到执行同步路由的线程，
        # 因此门面内部的提供者包装层会把阶段追加到它上面。
        # The route is matched inside the app, so the trace opens under the placeholder operation and is renamed once
        # scope['endpoint'] is known; stages appended meanwhile belong to the same record either way.
        # 路由在应用内部才匹配，因此追踪先以占位操作名打开，待 scope['endpoint'] 已知后再改名；
        # 期间追加的阶段无论如何都属于同一条记录。
        try:
            with trace(UNMATCHED_OPERATION, request_id) as record:
                try:
                    await self.app(scope, receive, send_with_id)
                except Exception as exc:  # noqa: BLE001 - the catch-all of the service face
                    # An exception no handler mapped (Starlette's plain-text 500 otherwise): the redacted
                    # INTERNAL_000 envelope with X-Request-ID, unless the response already started streaming.
                    # 没有任何处理器映射的异常（否则是 Starlette 的纯文本 500）：带 X-Request-ID 的脱敏 INTERNAL_000
                    # 信封，除非响应已经开始流式输出。
                    if started:
                        raise
                    record.error = f"{type(exc).__name__}: {exc}"
                    await error_response(exc, request_id=request_id)(scope, receive, send_with_id)
        finally:
            # trace() fills duration_ms / error on exit, so the registry sees the finished record even when an
            # unhandled exception is on its way to Starlette's ServerErrorMiddleware.
            # trace() 在退出时填好 duration_ms / error，因此即使有未处理异常正传向 Starlette 的
            # ServerErrorMiddleware，登记表看到的也是已结束的记录。
            record.operation = _operation_of(scope)
            self.metrics.observe(record)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the facade on shutdown when create_app() built it (an injected HMS stays open for its owner).
    在关闭时关闭由 create_app() 自建的门面（注入的 HMS 留给其所有者处理）。
    """
    yield
    if getattr(app.state, "owns_hms", False):
        app.state.hms.close()


def create_app(hms: HMS | None = None, *, config: dict | None = None) -> FastAPI:
    """Build the FastAPI app over an HMS facade (injected, or HMS(config) — the offline defaults when both are None).
    基于 HMS 门面构建 FastAPI 应用（注入的实例，或 HMS(config)——二者皆为 None 时为离线默认值）。

    Input / 输入：
        hms: An HMS instance to serve; None → HMS(config) is created and closed on shutdown.
        hms：要提供服务的 HMS 实例；None → 创建 HMS(config)，并在关闭时关闭它。

        config: HMS config dict (Mem0-shaped or HMS-shaped) used only when hms is None.
        config：仅在 hms 为 None 时使用的 HMS 配置字典（Mem0 形状或 HMS 形状）。

    Output / 输出：
        The FastAPI app; app.state carries hms, owns_hms, server_token (from $HMS_SERVER_TOKEN, read once here) and
        metrics (the RequestMetrics behind GET /metrics); the RequestTelemetry middleware wraps every request.
        FastAPI 应用；app.state 携带 hms、owns_hms、server_token（来自 $HMS_SERVER_TOKEN，仅在此读取一次）与
        metrics（GET /metrics 背后的 RequestMetrics）；RequestTelemetry 中间件包住每个请求。

    State effect / 状态影响：
        Opens the configured store when no HMS is injected; DependencyError (extra 'server') without fastapi.
        未注入 HMS 时打开所配置的存储；缺少 fastapi 时抛出 DependencyError（extra 'server'）。
    """
    if _IMPORT_ERROR is not None:
        raise DependencyError(
            "The 'fastapi' package is required for the HMS HTTP server",
            "DEPS_001",
            details={"package": "fastapi", "extra": "server"},
        ) from _IMPORT_ERROR
    facade = hms if hms is not None else HMS(config)
    app = FastAPI(title="HMS Elite", version=__version__, lifespan=_lifespan)
    app.state.hms = facade
    app.state.owns_hms = hms is None
    app.state.server_token = resolve_token()
    app.state.metrics = RequestMetrics()
    app.add_middleware(RequestTelemetry, metrics=app.state.metrics)
    app.add_api_route("/health", get_health, methods=["GET"], response_model=HealthResponse, name=get_health.__name__)
    router = APIRouter(dependencies=[Depends(require_bearer)], responses=_ERROR_RESPONSES)
    for method, path, handler, model, _operation in _ROUTES:
        router.add_api_route(path, handler, methods=[method], response_model=model, name=handler.__name__)
    app.include_router(router)
    # Starlette picks the handler by MRO: hms_elite ValidationError (an HMSError and a ValueError) hits the HMSError
    # handler; pydantic errors raised inside the facade are ValueErrors and reach the second one.
    # Starlette 按 MRO 选择处理器：hms_elite 的 ValidationError（既是 HMSError 也是 ValueError）命中 HMSError
    # 处理器；门面内部抛出的 pydantic 错误属于 ValueError，落到第二个处理器。
    app.add_exception_handler(HMSError, _handle_error)
    app.add_exception_handler(ValueError, _handle_error)
    # Any other exception gets the redacted INTERNAL_000 envelope from RequestTelemetry itself: Starlette routes an
    # `Exception` handler to its outermost ServerErrorMiddleware, outside the telemetry wrapper, so the response would
    # carry no X-Request-ID and the trace would not see the status.
    # 其余任何异常由 RequestTelemetry 自身给出脱敏的 INTERNAL_000 信封：Starlette 把 `Exception` 处理器路由到最外层的
    # ServerErrorMiddleware，在遥测包装之外，响应因此不带 X-Request-ID，追踪也看不到状态码。
    return app


def _import_uvicorn() -> Any:
    """Import uvicorn or raise DependencyError (extra 'server').
    导入 uvicorn，否则抛出 DependencyError（extra 'server'）。
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DependencyError(
            "The 'uvicorn' package is required to run the HMS HTTP server",
            "DEPS_001",
            details={"package": "uvicorn", "extra": "server"},
        ) from exc
    return uvicorn


def main(argv: list[str] | None = None) -> int:
    """Console script `hms-server`: parse --host/--port/--config/--log-level and serve create_app() under uvicorn.
    控制台脚本 `hms-server`：解析 --host/--port/--config/--log-level，并在 uvicorn 下运行 create_app()。

    Input / 输入：
        argv: Argument list (None → sys.argv[1:]); --config names a JSON file holding an HMS config dict.
        argv：参数列表（None → sys.argv[1:]）；--config 指向一个存放 HMS 配置字典的 JSON 文件。

    Output / 输出：
        Process exit code 0 after uvicorn returns; DependencyError (extra 'server') when uvicorn/fastapi is missing.
        uvicorn 返回后的进程退出码 0；缺少 uvicorn/fastapi 时抛出 DependencyError（extra 'server'）。

    State effect / 状态影响：
        Opens the store ($HMS_DIR/hms.db or the --config path) and binds the TCP port until interrupted.
        打开存储（$HMS_DIR/hms.db 或 --config 指定的路径）并绑定 TCP 端口，直到被中断。
    """
    parser = argparse.ArgumentParser(prog="hms-server", description="HMS Elite HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="JSON file holding an HMS config dict")
    parser.add_argument("--log-level", default="info", dest="log_level")
    ns = parser.parse_args(argv)
    uvicorn = _import_uvicorn()
    config = json.loads(pathlib.Path(ns.config).read_text(encoding="utf-8")) if ns.config else None
    uvicorn.run(create_app(config=config), host=ns.host, port=ns.port, log_level=ns.log_level)
    return 0


__all__ = [
    "RESET_MESSAGE",
    "UNMATCHED_OPERATION",
    "RequestTelemetry",
    "create_app",
    "error_response",
    "error_status",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
