"""Request-scoped dependencies of the HTTP service face: the HMS behind the app, the bearer check, the metrics registry.
HTTP 服务面的请求级依赖：应用背后的 HMS、Bearer 鉴权检查、指标登记表。
"""

# Purpose (EN): Keep the FastAPI glue (app.state lookups, Authorization parsing) out of the route handlers, and hold
#               the process-wide RequestMetrics that GET /metrics renders in the Prometheus text format.
# 目的（中）：把 FastAPI 胶水逻辑（app.state 查找、Authorization 解析）从路由处理函数中分离出来，
#            并持有 GET /metrics 以 Prometheus 文本格式渲染的进程级 RequestMetrics。

# Input (EN): A Starlette Request (app.state.hms / server_token / metrics, the Authorization header), bare values,
#             finished RequestTrace objects from the telemetry middleware.
# 输入（中）：一个 Starlette Request（app.state.hms / server_token / metrics、Authorization 头）、裸值、
#            来自遥测中间件的已结束 RequestTrace 对象。

# Output (EN): The HMS facade, the RequestMetrics registry, None after a successful bearer check (AuthError AUTH_001
#              when it fails), Prometheus exposition text.
# 输出（中）：HMS 门面、RequestMetrics 登记表、鉴权通过后的 None（失败时抛出 AuthError AUTH_001）、
#            Prometheus 暴露格式文本。

# State effect (EN): RequestMetrics.observe mutates only its own counters (under a lock); resolve_token reads
#                    HMS_SERVER_TOKEN from the environment; nothing here touches the store, graph or index.
# 状态影响（中）：RequestMetrics.observe 只在锁内修改自身计数器；resolve_token 读取环境变量 HMS_SERVER_TOKEN；
#                这里不触碰存储、图或索引。

from __future__ import annotations

import hmac
import os
import threading
from typing import Any

from hms_elite.exceptions import HMSError
from hms_elite.memory.main import HMS
from hms_elite.telemetry import STAGE_LLM_GENERATE, RequestTrace, current_trace, is_embedding_stage

# The Request type is only used in annotations; without fastapi the module stays importable and the name degrades
# to Any (from __future__ import annotations keeps every annotation a string until FastAPI resolves it).
# Request 类型只用于注解；没有 fastapi 时模块仍可导入，该名字退化为 Any
# （from __future__ import annotations 让每个注解保持为字符串，直到 FastAPI 解析它）。
try:
    from fastapi import Request
except ImportError:  # pragma: no cover - depends on the environment
    Request = Any  # type: ignore[misc,assignment]

# Environment variable that switches bearer authentication on; unset or empty means every request is accepted.
# 开启 Bearer 鉴权的环境变量；未设置或为空表示接受每个请求。
TOKEN_ENV = "HMS_SERVER_TOKEN"
BEARER_PREFIX = "bearer "
# Request-correlation header: honoured when the client sends it, generated (uuid4 hex) otherwise, always echoed back.
# 请求关联头：客户端发送时沿用，否则生成（uuid4 十六进制串），并且总是回写到响应。
REQUEST_ID_HEADER = "X-Request-ID"
# Longest client-supplied request id accepted verbatim; longer or empty values are replaced by a generated one.
# 原样接受的客户端请求 id 的最大长度；更长或为空的值会被生成的 id 替换。
REQUEST_ID_MAX_LEN = 128
# Upper edges (milliseconds) of the hms_request_duration_ms histogram; '+Inf' is appended when rendering.
# hms_request_duration_ms 直方图的桶上界（毫秒）；渲染时追加 '+Inf'。
DURATION_BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
# Prometheus text exposition content type (version 0.0.4 is the plain-text format every scraper accepts).
# Prometheus 文本暴露格式的内容类型（0.0.4 版是每个抓取器都接受的纯文本格式）。
METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class AuthError(HMSError):
    """Missing or wrong bearer token (mapped to HTTP 401 by the app).
    缺失或错误的 Bearer 令牌（由应用映射为 HTTP 401）。
    """

    default_code = "AUTH_001"
    default_suggestion = f"Send 'Authorization: Bearer <token>' matching the {TOKEN_ENV} environment variable"


def resolve_token(explicit: str | None = None) -> str | None:
    """The bearer token the server enforces: the explicit value, else $HMS_SERVER_TOKEN; empty → None (auth off).
    服务端强制使用的 Bearer 令牌：显式值，否则取 $HMS_SERVER_TOKEN；为空 → None（关闭鉴权）。

    Input / 输入：
        explicit: Token passed by the caller of create_app (wins over the environment).
        explicit：create_app 调用方传入的令牌（优先于环境变量）。

    Output / 输出：
        A non-empty token string, or None when authentication is disabled.
        非空令牌字符串；关闭鉴权时为 None。

    State effect / 状态影响：
        None (reads the environment).
        无（读取环境变量）。
    """
    token = explicit if explicit is not None else os.environ.get(TOKEN_ENV)
    return token or None


def check_bearer(authorization: str | None, token: str | None) -> None:
    """Validate an Authorization header against the enforced token (constant-time compare); no token → accept.
    按强制令牌校验 Authorization 头（常数时间比较）；没有令牌 → 直接接受。

    Input / 输入：
        authorization: Raw header value ('Bearer <token>', case-insensitive scheme) or None.
        authorization：原始头值（'Bearer <token>'，方案不区分大小写）或 None。

        token: The enforced token, or None when authentication is disabled.
        token：强制令牌；关闭鉴权时为 None。

    Output / 输出：
        None; AuthError (AUTH_001) when the header is missing, malformed or carries another token.
        无；头缺失、格式错误或令牌不匹配时抛出 AuthError（AUTH_001）。

    State effect / 状态影响：
        None.
        无。
    """
    if token is None:
        return
    header = authorization or ""
    if not header.lower().startswith(BEARER_PREFIX):
        raise AuthError("Missing bearer token", "AUTH_001", details={"header": "Authorization"})
    presented = header[len(BEARER_PREFIX) :].strip()
    if not hmac.compare_digest(presented.encode("utf-8"), token.encode("utf-8")):
        raise AuthError("Invalid bearer token", "AUTH_001", details={"header": "Authorization"})


def require_bearer(request: Request) -> None:
    """FastAPI dependency: enforce app.state.server_token on the request's Authorization header.
    FastAPI 依赖：用 app.state.server_token 校验请求的 Authorization 头。

    Input / 输入：
        request: The incoming request (headers + app.state).
        request：传入的请求（头 + app.state）。

    Output / 输出：
        None; AuthError propagates to the app's HMSError handler (401).
        无；AuthError 传播到应用的 HMSError 处理器（401）。

    State effect / 状态影响：
        None.
        无。
    """
    check_bearer(request.headers.get("authorization"), getattr(request.app.state, "server_token", None))


def get_hms(request: Request) -> HMS:
    """The HMS facade bound to the application (app.state.hms).
    绑定到应用的 HMS 门面（app.state.hms）。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        The HMS instance create_app stored on app.state.
        create_app 存放在 app.state 上的 HMS 实例。

    State effect / 状态影响：
        None.
        无。
    """
    return request.app.state.hms


def request_usage() -> dict[str, Any] | None:
    """Token usage this request has spent so far, read from the telemetry trace the middleware opened.
    本请求迄今消耗的 token 用量，读取自中间件打开的遥测追踪。

    Input / 输入：
        none (reads the context-local trace).
        无（读取上下文局部的追踪）。

    Output / 输出：
        RequestTrace.to_dict() while a traced request is running, else None.
        处于被追踪的请求中时为 RequestTrace.to_dict()，否则为 None。

    State effect / 状态影响：
        None (read-only).
        无（只读）。
    """
    record = current_trace()
    return None if record is None else record.to_dict()


def _label_value(value: str) -> str:
    """Escape a label value for the Prometheus text format (backslash, double quote, newline).
    为 Prometheus 文本格式转义标签值（反斜杠、双引号、换行）。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float | int) -> str:
    """Render a sample value: integers verbatim, floats in repr form (Prometheus accepts both).
    渲染样本值：整数原样，浮点数用 repr 形式（Prometheus 两者都接受）。
    """
    return str(int(value)) if float(value).is_integer() else repr(float(value))


class RequestMetrics:
    """Process-wide counters and one duration histogram fed by finished RequestTraces; renders Prometheus text.
    由已结束的 RequestTrace 喂入的进程级计数器与一个耗时直方图；渲染为 Prometheus 文本。
    """

    def __init__(self, buckets: tuple[float, ...] = DURATION_BUCKETS_MS) -> None:
        """Start every counter at zero with the given histogram bucket edges.
        以给定的直方图桶边界从零开始每个计数器。
        """
        self._lock = threading.Lock()
        self._buckets = tuple(sorted(float(edge) for edge in buckets))
        self._llm_calls = 0
        self._embedding_calls = 0
        # (direction, estimated) -> tokens; both label values are rendered as strings.
        # (方向, 是否估算) -> token 数；两个标签值都渲染为字符串。
        self._tokens: dict[tuple[str, bool], int] = {}
        # operation -> [bucket counts..., +Inf count], sum, count
        # 操作名 -> [各桶计数..., +Inf 计数]、总和、次数
        self._durations: dict[str, tuple[list[int], float, int]] = {}

    def observe(self, trace: RequestTrace) -> None:
        """Fold one finished trace in: its llm.generate / embedding.* stages and its wall time under its operation.
        折入一条已结束的追踪：它的 llm.generate / embedding.* 阶段，以及按操作名归类的墙钟耗时。

        Input / 输入：
            trace: The RequestTrace the telemetry middleware closed (duration_ms filled).
            trace：遥测中间件已关闭的 RequestTrace（duration_ms 已填好）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Increments the counters and the histogram under the lock.
            在锁内递增计数器与直方图。
        """
        with self._lock:
            for record in trace.stages:
                if record.operation == STAGE_LLM_GENERATE:
                    self._llm_calls += record.provider_calls
                    for direction, tokens in (("input", record.input_tokens), ("output", record.output_tokens)):
                        key = (direction, bool(record.estimated))
                        self._tokens[key] = self._tokens.get(key, 0) + int(tokens or 0)
                elif is_embedding_stage(record.operation):
                    self._embedding_calls += record.provider_calls
            counts, total, count = self._durations.get(trace.operation, ([0] * (len(self._buckets) + 1), 0.0, 0))
            counts = list(counts)
            for index, edge in enumerate(self._buckets):
                if trace.duration_ms <= edge:
                    counts[index] += 1
            counts[-1] += 1
            self._durations[trace.operation] = (counts, total + float(trace.duration_ms), count + 1)

    def reset(self) -> None:
        """Zero every counter and drop every histogram series.
        将所有计数器归零并丢弃所有直方图序列。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Replaces the internal state under the lock.
            在锁内替换内部状态。
        """
        with self._lock:
            self._llm_calls = 0
            self._embedding_calls = 0
            self._tokens = {}
            self._durations = {}

    def render(self) -> str:
        """Prometheus text exposition of the four metric families (HELP / TYPE lines plus one sample per series).
        四个指标族的 Prometheus 文本暴露格式（HELP / TYPE 行加每个序列一个样本）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Text ending in a newline: hms_llm_calls_total, hms_llm_tokens_total{direction,estimated},
            hms_embedding_calls_total, hms_request_duration_ms{operation} (histogram _bucket/_sum/_count).
            以换行结尾的文本：hms_llm_calls_total、hms_llm_tokens_total{direction,estimated}、
            hms_embedding_calls_total、hms_request_duration_ms{operation}（直方图 _bucket/_sum/_count）。

        State effect / 状态影响：
            None (snapshots the state under the lock).
            无（在锁内对状态做快照）。
        """
        with self._lock:
            llm_calls = self._llm_calls
            embedding_calls = self._embedding_calls
            tokens = dict(self._tokens)
            durations = {name: (list(counts), total, count) for name, (counts, total, count) in self._durations.items()}
        lines = [
            "# HELP hms_llm_calls_total LLM generate calls observed by the HTTP face.",
            "# TYPE hms_llm_calls_total counter",
            f"hms_llm_calls_total {llm_calls}",
            "# HELP hms_llm_tokens_total LLM tokens by direction; estimated=true when counted by estimate_tokens.",
            "# TYPE hms_llm_tokens_total counter",
        ]
        for (direction, estimated), value in sorted(tokens.items()):
            flag = "true" if estimated else "false"
            lines.append(f'hms_llm_tokens_total{{direction="{direction}",estimated="{flag}"}} {value}')
        lines += [
            "# HELP hms_embedding_calls_total Embedding calls (embed + embed_batch) observed by the HTTP face.",
            "# TYPE hms_embedding_calls_total counter",
            f"hms_embedding_calls_total {embedding_calls}",
            "# HELP hms_request_duration_ms Wall time of one HTTP request by operation, in milliseconds.",
            "# TYPE hms_request_duration_ms histogram",
        ]
        for operation, (counts, total, count) in sorted(durations.items()):
            label = _label_value(operation)
            for edge, bucket_count in zip((*self._buckets, float("inf")), counts):
                le = "+Inf" if edge == float("inf") else _fmt(edge)
                lines.append(f'hms_request_duration_ms_bucket{{operation="{label}",le="{le}"}} {bucket_count}')
            lines.append(f'hms_request_duration_ms_sum{{operation="{label}"}} {_fmt(total)}')
            lines.append(f'hms_request_duration_ms_count{{operation="{label}"}} {count}')
        return "\n".join(lines) + "\n"


def get_metrics(request: Request) -> RequestMetrics:
    """The RequestMetrics registry bound to the application (app.state.metrics).
    绑定到应用的 RequestMetrics 登记表（app.state.metrics）。

    Input / 输入：
        request: The incoming request.
        request：传入的请求。

    Output / 输出：
        The registry create_app stored on app.state.
        create_app 存放在 app.state 上的登记表。

    State effect / 状态影响：
        None.
        无。
    """
    return request.app.state.metrics


__all__ = [
    "BEARER_PREFIX",
    "DURATION_BUCKETS_MS",
    "METRICS_CONTENT_TYPE",
    "REQUEST_ID_HEADER",
    "REQUEST_ID_MAX_LEN",
    "TOKEN_ENV",
    "AuthError",
    "RequestMetrics",
    "check_bearer",
    "get_hms",
    "get_metrics",
    "require_bearer",
    "resolve_token",
]
