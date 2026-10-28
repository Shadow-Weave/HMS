"""Group S tests: the FastAPI service face over the offline default HMS (routes, envelopes, errors, auth, telemetry).
S 组测试：基于离线默认 HMS 的 FastAPI 服务面（路由、信封、错误映射、鉴权、遥测）。
"""

# Purpose (EN): Pin the HTTP surface: every endpoint's success path, the 400/401/404/500 mapping, bearer auth, the
#               `hms-server` entry point, GET /usage + GET /metrics and the X-Request-ID round trip; skipped as a whole
#               when fastapi (the `server` extra) is not installed.
# 目的（中）：钉住 HTTP 表面：每个端点的成功路径、400/401/404/500 映射、Bearer 鉴权、`hms-server` 入口、
#            GET /usage + GET /metrics 与 X-Request-ID 往返；未安装 fastapi（`server` extra）时整体跳过。

# Input (EN): fastapi.testclient.TestClient over create_app(); HMS_DIR is redirected to tmp_path by conftest.
# 输入（中）：基于 create_app() 的 fastapi.testclient.TestClient；conftest 已把 HMS_DIR 重定向到 tmp_path。

# Output (EN): pytest results.
# 输出（中）：pytest 结果。

# State effect (EN): Writes tmp_path/hms.db only; sockets stay blocked (TestClient is in-process ASGI).
# 状态影响（中）：只写 tmp_path/hms.db；套接字保持阻断（TestClient 是进程内 ASGI）。

from __future__ import annotations

import importlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
from typing import Any

import pytest

from hms_elite import HMS, __version__
from hms_elite.exceptions import DependencyError, StoreError, ValidationError
from hms_elite.models import Scope, StatePackage
from hms_elite.server import app as server_app
from hms_elite.server import create_app, main
from hms_elite.server.deps import (
    METRICS_CONTENT_TYPE,
    REQUEST_ID_HEADER,
    AuthError,
    RequestMetrics,
    check_bearer,
    resolve_token,
)
from hms_elite.server.schemas import merge_scope
from hms_elite.telemetry import RequestTrace, StageRecord, usage_of
from hms_elite.timeutil import next_day_midnight

# The real socket class, captured before conftest's no_network fixture swaps it out.
# 真实的 socket 类，在 conftest 的 no_network 夹具替换它之前捕获。
_RealSocket = socket.socket

ROOT = pathlib.Path(__file__).resolve().parents[1]
U = "alice"
RUN = "run-coffee"
SCOPE = {"user_id": U, "run_id": RUN}
LATTE = "I like oat-milk lattes in the morning."
LATTE_FACT = "User likes oat-milk lattes in the morning."
SARAH = "Sarah joined the billing team in March."
ERROR_KEYS = {"error_type", "error_code", "message", "details", "suggestion", "debug_info"}
EXPECTED_ROUTES = {
    ("POST", "/memories"),
    ("GET", "/memories"),
    ("GET", "/memories/{memory_id}"),
    ("POST", "/search"),
    ("PUT", "/memories/{memory_id}"),
    ("GET", "/memories/{memory_id}/history"),
    ("DELETE", "/memories/{memory_id}"),
    ("DELETE", "/memories"),
    ("POST", "/reset"),
    ("GET", "/health"),
    ("POST", "/events"),
    ("POST", "/compile"),
    ("GET", "/why/{target}"),
    ("POST", "/feedback"),
    ("POST", "/checkpoint"),
    ("POST", "/rollback"),
    ("POST", "/handoff"),
    ("GET", "/transitions"),
    ("POST", "/consolidate"),
    ("GET", "/stats"),
    ("GET", "/usage"),
    ("GET", "/metrics"),
}
USAGE_KEYS = {
    "llm_calls", "input_tokens", "output_tokens", "total_tokens", "estimated", "embedding_calls", "embedded_texts",
    "duration_ms",
}  # fmt: skip
# The simplest Prometheus text-format sample rule: `name{labels} value` (labels optional), one per line.
# 最简单的 Prometheus 文本格式样本规则：`name{labels} value`（labels 可选），每行一个。
SAMPLE_LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{(?:[A-Za-z_][A-Za-z0-9_]*="(?:[^"\\]|\\.)*",?)*\})? (\S+)$')
METRIC_FAMILIES = (
    "hms_llm_calls_total", "hms_llm_tokens_total", "hms_embedding_calls_total", "hms_request_duration_ms",
)  # fmt: skip


class _LocalPairSocket(_RealSocket):
    """socket.socket that admits AF_UNIX only (the event loop's self-pipe socketpair); network families still raise.
    只允许 AF_UNIX（事件循环自管道的 socketpair）的 socket.socket；网络地址族仍然抛错。
    """

    def __init__(self, family: int = -1, type: int = -1, proto: int = -1, fileno: Any = None) -> None:
        """Refuse every family but AF_UNIX unless HMS_TEST_ALLOW_NETWORK=1.
        除非 HMS_TEST_ALLOW_NETWORK=1，否则拒绝 AF_UNIX 之外的所有地址族。
        """
        if family != socket.AF_UNIX and os.environ.get("HMS_TEST_ALLOW_NETWORK") != "1":
            raise RuntimeError("network access is disabled in tests (set HMS_TEST_ALLOW_NETWORK=1 to allow)")
        super().__init__(family, type, proto, fileno)


@pytest.fixture(autouse=True)
def local_pairs_only(no_network: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Narrow conftest's socket block: TestClient runs an asyncio loop whose self-pipe is an AF_UNIX socketpair.
    收窄 conftest 的套接字阻断：TestClient 运行的 asyncio 循环，其自管道是一个 AF_UNIX socketpair。

    Input / 输入：
        no_network: conftest's blanket guard (ordered first so this override wins).
        no_network：conftest 的全面守卫（先执行，使本覆盖生效）。

        monkeypatch: pytest monkeypatch.
        monkeypatch：pytest 的 monkeypatch。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        socket.socket is _LocalPairSocket for the test; AF_INET/AF_INET6 sockets keep raising RuntimeError.
        测试期间 socket.socket 为 _LocalPairSocket；AF_INET/AF_INET6 套接字继续抛出 RuntimeError。
    """
    monkeypatch.setattr(socket, "socket", _LocalPairSocket)


def _import_test_client() -> Any:
    """Import fastapi.testclient.TestClient or skip: ImportError (no fastapi) and RuntimeError (starlette without httpx).
    导入 fastapi.testclient.TestClient，否则跳过：ImportError（无 fastapi）与 RuntimeError（starlette 缺 httpx）。
    """
    try:
        return importlib.import_module("fastapi.testclient").TestClient
    except (ImportError, RuntimeError) as exc:
        # starlette >= 1.6 raises RuntimeError (not ImportError) when neither httpx2 nor httpx is installed; the
        # `test` extra carries httpx so a `[server,test]` install runs these tests instead of skipping them.
        # starlette >= 1.6 在 httpx2 与 httpx 都缺失时抛 RuntimeError（而非 ImportError）；`test` extra 带上 httpx，
        # 使 `[server,test]` 安装能运行这些测试而不是跳过。
        pytest.skip(f"fastapi.testclient unavailable ({type(exc).__name__}: {exc})")


@pytest.fixture(autouse=True)
def _needs_fastapi() -> None:
    """Skip every test of this module without the `server` extra; hms_elite.server itself imports without it (guarded).
    缺少 `server` extra 时跳过本模块的每个测试；hms_elite.server 本身没有它也能导入（有守卫）。
    """
    # Fixture-level (not module-level) so that importing this module never raises Skip: test_bilingual_docstrings
    # imports every tests module and must keep running on machines without fastapi.
    # 放在夹具层（而非模块层），使导入本模块永不抛出 Skip：test_bilingual_docstrings 会导入每个 tests 模块，
    # 在没有 fastapi 的机器上也必须继续运行。
    _import_test_client()


@pytest.fixture
def test_client_cls() -> Any:
    """The fastapi.testclient.TestClient class, resolved lazily.
    惰性解析的 fastapi.testclient.TestClient 类。

    Input / 输入：
        none.
        无。

    Output / 输出：
        The TestClient class (skips the test when fastapi or its http client is missing).
        TestClient 类（缺少 fastapi 或其 http 客户端时跳过测试）。

    State effect / 状态影响：
        None.
        无。
    """
    return _import_test_client()


@pytest.fixture
def server_hms() -> Any:
    """The offline default HMS() (rule LLM + hash embedder + sqlite index) on tmp_path/hms.db, closed afterwards.
    位于 tmp_path/hms.db 的离线默认 HMS()（规则 LLM + 哈希嵌入器 + sqlite 索引），用后关闭。

    Input / 输入：
        none.
        无。

    Output / 输出：
        An HMS instance (generator fixture).
        一个 HMS 实例（生成器夹具）。

    State effect / 状态影响：
        Creates tmp_path/hms.db; closes the store on teardown.
        创建 tmp_path/hms.db；拆卸时关闭存储。
    """
    facade = HMS()
    yield facade
    facade.close()


@pytest.fixture
def client(server_hms: HMS, test_client_cls: Any) -> Any:
    """A TestClient over create_app(server_hms) with the lifespan running (no bearer token in the environment).
    基于 create_app(server_hms)、运行着 lifespan 的 TestClient（环境中没有 Bearer 令牌）。

    Input / 输入：
        server_hms: The offline HMS fixture.
        server_hms：离线 HMS 夹具。

        test_client_cls: The TestClient class fixture.
        test_client_cls：TestClient 类夹具。

    Output / 输出：
        A TestClient (generator fixture).
        一个 TestClient（生成器夹具）。

    State effect / 状态影响：
        None beyond the HMS fixture.
        除 HMS 夹具外无副作用。
    """
    with test_client_cls(create_app(server_hms)) as test_client:
        yield test_client


def _add_latte(client: Any) -> str:
    """POST the latte sentence into SCOPE and return its Mem0 id.
    把 latte 句子 POST 到 SCOPE 中并返回其 Mem0 id。
    """
    response = client.post("/memories", json={"messages": LATTE, "timestamp": "2026-04-02T08:00:00Z", **SCOPE})
    assert response.status_code == 200, response.text
    return response.json()["results"][0]["id"]


def _error(response: Any, status: int, code: str) -> dict:
    """Assert an error response's status and code and return its body.
    断言错误响应的状态码与错误码并返回响应体。
    """
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == ERROR_KEYS and body["error_code"] == code, body
    return body


def test_health_and_route_table(client: Any) -> None:
    """GET /health carries the version and the OpenAPI path table is exactly the Mem0 + HMS-native route set.
    GET /health 携带版本号，且 OpenAPI 路径表恰好等于 Mem0 + HMS 原生的路由集合。

    Input / 输入：
        client: The TestClient fixture.
        client：TestClient 夹具。

    Output / 输出：
        Assertions on /health, /openapi.json and the documented error responses.
        对 /health、/openapi.json 与记录的错误响应的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert client.get("/health").json() == {
        "status": "ok",
        "version": __version__,
        "providers": {"llm": "rule", "embedder": "hash", "vector_store": "sqlite"},
    }
    spec = client.get("/openapi.json").json()
    routes = {(method.upper(), path) for path, methods in spec["paths"].items() for method in methods}
    assert routes == EXPECTED_ROUTES
    documented = set(spec["paths"]["/memories"]["post"]["responses"])
    assert {"200", "400", "401", "404", "500", "422"} <= documented
    assert set(spec["paths"]["/health"]["get"]["responses"]) == {"200"}


def test_mem0_endpoints_roundtrip(client: Any) -> None:
    """The nine Mem0 routes return the facade envelopes verbatim; a missing lineage is 404; reset empties the store.
    九条 Mem0 路由逐字返回门面信封；谱系不存在为 404；reset 清空存储。

    Input / 输入：
        client: The TestClient fixture.
        client：TestClient 夹具。

    Output / 输出：
        Assertions on every Mem0-shaped response.
        对每个 Mem0 形状响应的断言。

    State effect / 状态影响：
        Writes then resets tmp_path/hms.db.
        写入并随后重置 tmp_path/hms.db。
    """
    added = client.post("/memories", json={"messages": LATTE, "timestamp": "2026-04-02T08:00:00Z", **SCOPE}).json()
    assert set(added) == {"results", "usage"} and len(added["results"]) == 1
    # The add route spends tokens, so its usage must report the extraction LLM calls and the embeddings, split apart.
    # add 路由会消耗 token，因此其 usage 必须分开报告抽取用的 LLM 调用与嵌入。
    added_usage = added["usage"]
    assert set(added_usage) == {"request_id", "llm_input_tokens", "llm_output_tokens", "embedding_tokens",
                                "total_tokens", "estimated", "stages"}
    assert added_usage["llm_input_tokens"] > 0 and added_usage["llm_output_tokens"] > 0
    assert added_usage["embedding_tokens"] > 0 and added_usage["estimated"] is True
    assert added_usage["total_tokens"] == (
        added_usage["llm_input_tokens"] + added_usage["llm_output_tokens"] + added_usage["embedding_tokens"]
    )
    stage_ops = {stage["operation"] for stage in added_usage["stages"]}
    assert "llm.generate" in stage_ops and any(op.startswith("embedding.") for op in stage_ops)
    row = added["results"][0]
    assert row["memory"] == LATTE_FACT and row["event"] == "ADD" and row["version"] == 1 and row["anchor_ids"]
    mem = row["id"]
    listed = client.get("/memories", params={"user_id": U}).json()
    # Scope guard: only the token-consuming routes report usage, so a plain listing must not carry the key.
    # 范围守卫：只有耗 token 的路由才报告 usage，因此纯列举不带该键。
    assert set(listed) == {"results"}
    assert [r["id"] for r in listed["results"]] == [mem] and "score" not in listed["results"][0]
    assert client.get(f"/memories/{mem}").json()["memory"] == LATTE_FACT
    _error(client.get("/memories/mem_missing"), 404, "NOT_FOUND_001")
    found = client.post("/search", json={"query": "oat-milk lattes", "user_id": U, "threshold": 0.0}).json()
    assert found["results"][0]["id"] == mem and found["results"][0]["score"] > 0
    explained = client.post(
        "/search", json={"query": "lattes", "filters": {"user_id": U}, "threshold": 0.0, "explain": True}
    ).json()
    assert "score_details" in explained["results"][0]
    updated = client.put(f"/memories/{mem}", json={"text": "User likes flat whites."}).json()
    assert set(updated) == {"message", "usage"} and updated["message"] == "Memory updated successfully!"
    # Rewriting the text re-embeds the new version, so the embedding count must be non-zero and the LLM count zero.
    # 改写文本会重新嵌入新版本，因此嵌入数为正、LLM 数为 0。
    assert updated["usage"]["embedding_tokens"] > 0 and updated["usage"]["llm_input_tokens"] == 0
    assert client.get(f"/memories/{mem}").json()["memory"] == "User likes flat whites."
    history = client.get(f"/memories/{mem}/history").json()
    assert [h["event"] for h in history] == ["ADD", "UPDATE"] and history[0]["memory_id"] == mem
    assert client.get("/memories/mem_missing/history").json() == []
    assert client.delete(f"/memories/{mem}").json() == {"message": "Memory deleted successfully!"}
    _error(client.get(f"/memories/{mem}"), 404, "NOT_FOUND_001")
    _error(client.delete(f"/memories/{mem}"), 404, "NOT_FOUND_001")
    # a list-of-messages body and infer=False store raw message atoms without any LLM call
    # 消息列表请求体加 infer=False 会直接存原始消息原子，不调用 LLM
    raw = client.post(
        "/memories",
        json={"messages": [{"role": "user", "content": SARAH}], "infer": False, "user_id": U, "run_id": "run-a"},
    ).json()
    assert raw["results"][0]["memory"] == SARAH and raw["results"][0]["role"] == "user"
    assert client.delete("/memories", params={"user_id": U, "run_id": "run-a"}).json() == {
        "message": "Memories deleted successfully!"
    }
    assert client.get("/memories", params={"user_id": U, "run_id": "run-a"}).json() == {"results": []}
    _add_latte(client)
    assert client.post("/reset").json() == {"message": "All memories reset"}
    assert client.get("/memories", params={"user_id": U}).json() == {"results": []}


def test_update_expiration_absent_vs_null(client: Any) -> None:
    """PUT without expiration_date leaves valid_to untouched; PUT with null clears it; an empty body is 400.
    不带 expiration_date 的 PUT 不改变 valid_to；带 null 的 PUT 清除它；空请求体为 400。

    Input / 输入：
        client: The TestClient fixture.
        client：TestClient 夹具。

    Output / 输出：
        Assertions on valid_to across three PUTs (the facade's _UNSET sentinel survives the JSON boundary).
        对三次 PUT 之后 valid_to 的断言（门面的 _UNSET 哨兵跨过 JSON 边界依然有效）。

    State effect / 状态影响：
        Writes tmp_path/hms.db.
        写入 tmp_path/hms.db。
    """
    added = client.post("/memories", json={"messages": LATTE, "expiration_date": "2030-01-01", **SCOPE}).json()
    mem = added["results"][0]["id"]
    expected = next_day_midnight("2030-01-01")
    assert added["results"][0]["valid_to"] == expected
    assert client.put(f"/memories/{mem}", json={"metadata": {"k": 1}}).status_code == 200
    row = client.get(f"/memories/{mem}").json()
    assert row["valid_to"] == expected and row["metadata"] == {"k": 1}
    assert client.put(f"/memories/{mem}", json={"expiration_date": None}).status_code == 200
    assert client.get(f"/memories/{mem}").json()["valid_to"] is None
    _error(client.put(f"/memories/{mem}", json={}), 400, "VALIDATION_011")


def test_error_mapping(client: Any, server_hms: HMS, monkeypatch: pytest.MonkeyPatch) -> None:
    """ValidationError → 400, NOT_FOUND_* → 404, other HMSError → 500, bare ValueError → 400; body = to_dict().
    ValidationError → 400、NOT_FOUND_* → 404、其余 HMSError → 500、裸 ValueError → 400；响应体 = to_dict()。

    Input / 输入：
        client: The TestClient fixture.
        client：TestClient 夹具。

        server_hms: The HMS behind the client (methods are monkeypatched to raise).
        server_hms：client 背后的 HMS（通过 monkeypatch 让方法抛错）。

        monkeypatch: pytest monkeypatch.
        monkeypatch：pytest 的 monkeypatch。

    Output / 输出：
        Assertions on status codes and error bodies; FastAPI's own 422 for a malformed body stays untouched.
        对状态码与错误体的断言；请求体格式错误时 FastAPI 自身的 422 保持不变。

    State effect / 状态影响：
        None beyond the fixtures.
        除夹具外无副作用。
    """
    body = _error(client.get("/memories"), 400, "VALIDATION_001")
    assert body["error_type"] == "ValidationError" and body["details"] and body["debug_info"] == {}
    bad_time = {"messages": LATTE, "timestamp": "not-a-date", **SCOPE}
    _error(client.post("/memories", json=bad_time), 400, "VALIDATION_004")
    _error(client.post("/memories", json={"messages": LATTE}), 400, "VALIDATION_001")
    _error(client.put("/memories/mem_missing", json={"text": "x"}), 404, "NOT_FOUND_001")
    _error(client.post("/rollback", json={"checkpoint_id": "ckp_missing"}), 404, "NOT_FOUND_001")
    _error(client.get("/why/nothing-here"), 404, "NOT_FOUND_001")
    _error(client.post("/compile", json={"query": "coffee", "mode": "zzz", **SCOPE}), 400, "VALIDATION_013")
    _error(client.post("/events", json={"events": [{"object_type": "bogus"}], "user_id": U}), 400, "VALIDATION_005")
    _error(client.get("/transitions", params={"user_id": U, "fmt": "xml"}), 400, "VALIDATION_013")
    _error(client.delete("/memories"), 400, "VALIDATION_001")
    assert client.post("/memories", json={"messages": LATTE, "bogus": 1, **SCOPE}).status_code == 422
    assert client.post("/search", json={"user_id": U}).status_code == 422

    def boom(**_: Any) -> dict:
        """Raise a non-validation HMSError.
        抛出一个非校验类 HMSError。
        """
        raise StoreError("disk on fire", details={"table": "memory_atoms"})

    monkeypatch.setattr(server_hms, "graph_stats", boom)
    body = _error(client.get("/stats"), 500, "STORE_001")
    # A 500 keeps the type and code; message and details are redacted (see test_internal_errors_are_redacted).
    # 500 保留类型与错误码；消息与细节被脱敏（见 test_internal_errors_are_redacted）。
    assert body["error_type"] == "StoreError" and body["details"] == {} and body["message"] == "internal error"
    monkeypatch.setattr(server_hms, "history", lambda memory_id: (_ for _ in ()).throw(ValueError("bare")))
    body = _error(client.get("/memories/x/history"), 400, "VALIDATION_000")
    assert body["message"] == "bare" and body["details"] == {"exception": "ValueError"}
    monkeypatch.setattr(server_hms, "get_all", lambda **_: Scope(bogus=1))
    body = _error(client.get("/memories", params={"user_id": U}), 400, "VALIDATION_000")
    assert body["details"] == {"exception": "ValidationError"} and "bogus" in body["message"]
    assert server_app.error_status(DependencyError("x")) == 500
    assert server_app.error_status(AuthError("x")) == 401
    assert server_app.error_status(ValidationError("x", "NOT_FOUND_001")) == 404


def test_internal_errors_are_redacted(
    client: Any, server_hms: HMS, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 500 never carries a StoreError's path / SQL; a non-HMS exception gets the INTERNAL_000 envelope with X-Request-ID.
    500 绝不携带 StoreError 的路径 / SQL；非 HMS 异常得到带 X-Request-ID 的 INTERNAL_000 信封。

    Starlette's plain-text 500 (no request id, no JSON) is never reached.
    绝不会落到 Starlette 的纯文本 500（无请求 id、无 JSON）。

    Input / 输入：
        client, server_hms, monkeypatch, caplog: fixtures (facade methods are monkeypatched to raise).
        client、server_hms、monkeypatch、caplog：夹具（通过 monkeypatch 让门面方法抛错）。

    Output / 输出：
        Assertions on the bodies, headers, content type and the server-side log line carrying the full error.
        对响应体、响应头、内容类型以及携带完整错误的服务端日志行的断言。

    State effect / 状态影响：
        None beyond the fixtures.
        除夹具外无副作用。
    """
    import logging

    secret_path = "/Users/someone/secret/hms.db"

    def leak(**_: Any) -> dict:
        """Raise a StoreError shaped like the ones storage.py builds (path in the message and the details).
        抛出一个与 storage.py 构造的形状相同的 StoreError（消息与细节里都有路径）。
        """
        raise StoreError(
            f"cannot open database '{secret_path}'",
            details={"path": secret_path, "sql": "SELECT secret FROM memory_atoms WHERE atom_id = ?"},
            suggestion="check the path",
        )

    monkeypatch.setattr(server_hms, "graph_stats", leak)
    with caplog.at_level(logging.ERROR, logger="hms_elite.server.app"):
        response = client.get("/stats", headers={"X-Request-ID": "req-redact"})
    body = _error(response, 500, "STORE_001")
    assert body["error_type"] == "StoreError" and body["message"] == "internal error"
    assert body["details"] == {} and body["suggestion"] is None and body["debug_info"] == {}
    assert secret_path not in response.text and "SELECT" not in response.text
    assert response.headers["X-Request-ID"] == "req-redact"
    logged = [r for r in caplog.records if "req-redact" in r.getMessage()]
    assert logged and secret_path in logged[0].getMessage() and logged[0].exc_info is not None

    for exc in (RuntimeError("boom"), KeyError("missing")):
        monkeypatch.setattr(server_hms, "graph_stats", lambda exc=exc, **_: (_ for _ in ()).throw(exc))
        response = client.get("/stats", headers={"X-Request-ID": f"req-{type(exc).__name__}"})
        body = _error(response, 500, "INTERNAL_000")
        assert response.headers["content-type"].startswith("application/json")
        assert response.headers["X-Request-ID"] == f"req-{type(exc).__name__}"
        assert body["error_type"] == "HMSError" and body["message"] == "internal error" and body["details"] == {}
        assert "boom" not in response.text and "missing" not in response.text


def test_native_endpoints_flow(client: Any) -> None:
    """Every HMS-native route end to end: events, compile, why, feedback, checkpoint, rollback, handoff, transitions.
    每条 HMS 原生路由端到端：events、compile、why、feedback、checkpoint、rollback、handoff、transitions。

    Input / 输入：
        client: The TestClient fixture.
        client：TestClient 夹具。

    Output / 输出：
        Assertions on every HMS-native response shape.
        对每个 HMS 原生响应形状的断言。

    State effect / 状态影响：
        Writes tmp_path/hms.db.
        写入 tmp_path/hms.db。
    """
    mem = _add_latte(client)
    events = client.post(
        "/events",
        json={"events": [{"object_type": "message", "role": "user", "payload": {"text": SARAH}}], **SCOPE},
    ).json()
    assert set(events) == {"results", "events", "eps", "edges", "constraints", "usage"}
    assert events["events"][0]["object_type"] == "message" and events["events"][0]["anchor_id"].startswith("anc_")
    compiled = client.post("/compile", json={"query": "coffee", "mode": "fast", **SCOPE}).json()
    assert set(compiled) == {"context", "card", "usage"}
    # The compiler itself calls no LLM, so compiling costs nothing but the query embedding.
    # 编译器自身不调用 LLM，因此编译的花费只有查询嵌入。
    assert compiled["usage"]["llm_input_tokens"] == 0 and compiled["usage"]["llm_output_tokens"] == 0
    assert compiled["card"].startswith("# Work State Card\n") and "Mode: fast" in compiled["card"]
    assert compiled["context"]["mode"] == "fast" and compiled["context"]["scope"] == {**SCOPE, "agent_id": None}
    assert LATTE_FACT in compiled["card"]
    why = client.get(f"/why/{mem}", params={"hops": 1}).json()
    assert why["target"]["kind"] == "atom" and why["target"]["resolved_from"] == "id"
    question = client.get("/why/Sarah", params={**SCOPE, "edge_types": "revision,causal"}).json()
    assert question["target"]["resolved_from"] == "query" and question["target"]["text"] == SARAH
    checkpoint = client.post("/checkpoint", json={"label": "before-feedback", **SCOPE}).json()
    assert set(checkpoint) == {"checkpoint_id"} and checkpoint["checkpoint_id"].startswith("ckp_")
    feedback = client.post(
        "/feedback", json={"target_id": mem, "feedback": "state_wrong", "correction": "User likes cortados."}
    ).json()
    assert feedback["message"] == "Feedback applied" and [a["action"] for a in feedback["actions"]] == [
        "archive", "create", "revision_edge",
    ]  # fmt: skip
    assert client.get(f"/memories/{mem}").json()["memory"] == "User likes cortados."
    rollback = client.post("/rollback", json={"checkpoint_id": checkpoint["checkpoint_id"], "reason": "undo"}).json()
    assert rollback["message"] == "Rolled back to before-feedback" and rollback["restored"]
    assert client.get(f"/memories/{mem}").json()["memory"] == LATTE_FACT
    handoff = client.post("/handoff", json={"goal": "coffee", **SCOPE})
    package = StatePackage.model_validate(handoff.json())
    assert package.goal == "coffee" and package.checkpoint_id.startswith("ckp_") and package.pcir.ok()
    assert client.post("/handoff", json={"checkpoint": False, **SCOPE}).json()["checkpoint_id"] is None
    tuples = client.get("/transitions", params={"user_id": U}).json()
    records = client.get("/transitions", params={"user_id": U, "fmt": "records"}).json()
    assert len(tuples) == len(records) >= 2 and all(isinstance(t, list) for t in tuples)
    assert records[0]["source_kind"] in {"revision", "rollback"}
    jsonl = client.get("/transitions", params={"user_id": U, "fmt": "jsonl", "kinds": "rollback"})
    assert jsonl.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in jsonl.text.split("\n")]
    assert lines and all(line["source_kind"] == "rollback" for line in lines)
    report = client.post("/consolidate", json={"user_id": U}).json()
    assert set(report["phases"]) == {"orient", "collect", "consolidate", "prune"} and report["anchor_id"]
    stats = client.get("/stats", params=SCOPE).json()
    assert {"nodes", "edges", "valid_edges", "transitions", "witness_coverage"} <= set(stats)
    assert client.get("/stats").json()["transitions"] >= stats["transitions"] >= 2


def test_bearer_token_auth(server_hms: HMS, monkeypatch: pytest.MonkeyPatch, test_client_cls: Any) -> None:
    """HMS_SERVER_TOKEN set at create_app time gates every route but /health with a 401 AUTH_001 body.
    create_app 时设置的 HMS_SERVER_TOKEN 以 401 AUTH_001 响应体守住除 /health 外的每条路由。

    Input / 输入：
        server_hms: The offline HMS fixture.
        server_hms：离线 HMS 夹具。

        monkeypatch: pytest monkeypatch (environment).
        monkeypatch：pytest 的 monkeypatch（环境变量）。

        test_client_cls: The TestClient class fixture.
        test_client_cls：TestClient 类夹具。

    Output / 输出：
        Assertions on 401/200 per header and on the pure helpers check_bearer / resolve_token.
        对不同请求头下 401/200 的断言，以及对纯辅助函数 check_bearer / resolve_token 的断言。

    State effect / 状态影响：
        None beyond the fixtures.
        除夹具外无副作用。
    """
    monkeypatch.setenv("HMS_SERVER_TOKEN", "s3cret")
    app = create_app(server_hms)
    assert app.state.server_token == "s3cret"
    # the token is read once by create_app, not per request
    # 令牌由 create_app 读取一次，而不是每个请求读取
    monkeypatch.setenv("HMS_SERVER_TOKEN", "rotated")
    with test_client_cls(app) as client:
        assert client.get("/health").status_code == 200
        body = _error(client.get("/stats"), 401, "AUTH_001")
        assert body["error_type"] == "AuthError" and body["message"] == "Missing bearer token"
        _error(client.get("/stats", headers={"Authorization": "Bearer wrong"}), 401, "AUTH_001")
        _error(client.get("/stats", headers={"Authorization": "Basic s3cret"}), 401, "AUTH_001")
        _error(client.post("/memories", json={"messages": LATTE, **SCOPE}), 401, "AUTH_001")
        _error(client.get("/usage"), 401, "AUTH_001")
        _error(client.get("/metrics"), 401, "AUTH_001")
        assert client.get("/stats", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/stats", headers={"Authorization": "bearer s3cret"}).status_code == 200
    monkeypatch.delenv("HMS_SERVER_TOKEN")
    assert create_app(server_hms).state.server_token is None
    monkeypatch.setenv("HMS_SERVER_TOKEN", "")
    assert resolve_token() is None and resolve_token("explicit") == "explicit"
    check_bearer(None, None)
    check_bearer("Bearer x", "x")
    with pytest.raises(AuthError) as missing:
        check_bearer(None, "x")
    assert missing.value.error_code == "AUTH_001" and "HMS_SERVER_TOKEN" in missing.value.suggestion
    with pytest.raises(AuthError):
        check_bearer("Bearer xx", "x")


def _parse_metrics(text: str) -> dict[tuple[str, str], float]:
    """Parse Prometheus text: every non-comment line must match SAMPLE_LINE; returns {(name, labels): value}.
    解析 Prometheus 文本：每个非注释行必须匹配 SAMPLE_LINE；返回 {(名称, 标签串): 值}。
    """
    samples: dict[tuple[str, str], float] = {}
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_LINE.match(line)
        assert match, f"unparseable metrics line: {line!r}"
        name, labels, value = match.group(1), match.group(2) or "", match.group(3)
        samples[(name, labels)] = float("inf") if value == "+Inf" else float(value)
    return samples


def test_usage_and_metrics(client: Any, server_hms: HMS) -> None:
    """After one add + one search, /usage is non-zero and equals usage_of(hms); /metrics parses line by line.
    一次 add + 一次 search 之后，/usage 非零且等于 usage_of(hms)；/metrics 能逐行解析。

    Input / 输入：
        client: The TestClient fixture.
        client：TestClient 夹具。

        server_hms: The HMS behind the client (its meters are the source of /usage).
        server_hms：client 背后的 HMS（其计量器是 /usage 的来源）。

    Output / 输出：
        Assertions on the Usage JSON, the Prometheus text (HELP/TYPE per family, sample rule, histogram consistency,
        token totals matching /usage) and the content type.
        对 Usage JSON、Prometheus 文本（每族 HELP/TYPE、样本规则、直方图一致性、token 总数等于 /usage）
        与内容类型的断言。

    State effect / 状态影响：
        Writes tmp_path/hms.db.
        写入 tmp_path/hms.db。
    """
    _add_latte(client)
    found = client.post("/search", json={"query": "oat-milk lattes", "user_id": U, "threshold": 0.0}).json()
    assert found["results"]
    usage = client.get("/usage").json()
    assert set(usage) == USAGE_KEYS and usage == usage_of(server_hms).to_dict() == server_hms.usage().to_dict()
    assert usage["llm_calls"] >= 1 and usage["total_tokens"] > 0 and usage["embedding_calls"] >= 2
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"] and usage["estimated"] is True
    metrics = client.get("/metrics")
    assert metrics.status_code == 200 and metrics.headers["content-type"] == METRICS_CONTENT_TYPE
    text = metrics.text
    for family in METRIC_FAMILIES:
        assert f"# HELP {family} " in text and f"# TYPE {family} " in text
    samples = _parse_metrics(text)
    assert samples[("hms_llm_calls_total", "")] == usage["llm_calls"]
    assert samples[("hms_embedding_calls_total", "")] == usage["embedding_calls"]
    input_tokens = sum(v for (n, labels), v in samples.items() if n == "hms_llm_tokens_total" and "input" in labels)
    output_tokens = sum(v for (n, labels), v in samples.items() if n == "hms_llm_tokens_total" and "output" in labels)
    assert (input_tokens, output_tokens) == (usage["input_tokens"], usage["output_tokens"])
    assert ("hms_llm_tokens_total", '{direction="input",estimated="true"}') in samples
    for operation in ("add", "search", "usage"):
        count = samples[("hms_request_duration_ms_count", f'{{operation="{operation}"}}')]
        inf = samples[("hms_request_duration_ms_bucket", f'{{operation="{operation}",le="+Inf"}}')]
        assert count == inf >= 1 and samples[("hms_request_duration_ms_sum", f'{{operation="{operation}"}}')] > 0
    # the scrape itself is not yet in the text it returned; the next scrape counts it
    # 抓取本身还不在它返回的文本里；下一次抓取才把它计入
    assert ("hms_request_duration_ms_count", '{operation="metrics"}') not in samples
    assert _parse_metrics(client.get("/metrics").text)[("hms_request_duration_ms_count", '{operation="metrics"}')] == 1
    # an unmatched path is labelled `unmatched`; a 404 from a matched route keeps its operation
    # 未匹配路径标为 `unmatched`；已匹配路由返回的 404 保留其操作名
    assert client.get("/nope").status_code == 404
    _error(client.get("/memories/mem_missing"), 404, "NOT_FOUND_001")
    samples = _parse_metrics(client.get("/metrics").text)
    assert samples[("hms_request_duration_ms_count", '{operation="unmatched"}')] == 1
    assert samples[("hms_request_duration_ms_count", '{operation="get"}')] == 1
    # the registry escapes label values and survives a reset
    # 登记表会转义标签值，并能在重置后继续工作
    registry = RequestMetrics()
    record = RequestTrace(request_id="r", operation='we"ird\\op', duration_ms=3.0)
    record.add_stage(StageRecord(operation="llm.generate", input_tokens=5, output_tokens=2, total_tokens=7))
    registry.observe(record)
    rendered = _parse_metrics(registry.render())
    assert rendered[("hms_request_duration_ms_count", '{operation="we\\"ird\\\\op"}')] == 1
    assert rendered[("hms_request_duration_ms_bucket", '{operation="we\\"ird\\\\op",le="1"}')] == 0
    assert rendered[("hms_request_duration_ms_bucket", '{operation="we\\"ird\\\\op",le="5"}')] == 1
    assert rendered[("hms_llm_tokens_total", '{direction="input",estimated="false"}')] == 5
    registry.reset()
    assert _parse_metrics(registry.render())[("hms_llm_calls_total", "")] == 0


def test_request_id_round_trip(client: Any) -> None:
    """X-Request-ID is echoed when sent, generated (uuid4 hex) otherwise, and present on error and 404 responses.
    X-Request-ID 发送时原样回显，否则生成（uuid4 十六进制串），并且错误与 404 响应上也有。

    Input / 输入：
        client: The TestClient fixture.
        client：TestClient 夹具。

    Output / 输出：
        Assertions on the response header across success, error, unmatched and over-long-id requests.
        对成功、错误、未匹配与超长 id 请求的响应头断言。

    State effect / 状态影响：
        None beyond the fixtures.
        除夹具外无副作用。
    """
    echoed = client.get("/health", headers={REQUEST_ID_HEADER: "req-abc-123"})
    assert echoed.status_code == 200 and echoed.headers[REQUEST_ID_HEADER] == "req-abc-123"
    generated = client.get("/health").headers[REQUEST_ID_HEADER]
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    assert client.get("/health").headers[REQUEST_ID_HEADER] != generated
    error = client.get("/memories", headers={"x-request-id": "err-1"})
    assert error.status_code == 400 and error.headers[REQUEST_ID_HEADER] == "err-1"
    assert client.get("/nope", headers={REQUEST_ID_HEADER: "gone"}).headers[REQUEST_ID_HEADER] == "gone"
    assert re.fullmatch(r"[0-9a-f]{32}", client.get("/health", headers={REQUEST_ID_HEADER: "x" * 129}).headers[REQUEST_ID_HEADER])
    assert re.fullmatch(r"[0-9a-f]{32}", client.get("/health", headers={REQUEST_ID_HEADER: "  "}).headers[REQUEST_ID_HEADER])


def test_create_app_default_hms_and_lifespan(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, test_client_cls: Any
) -> None:
    """create_app() builds the offline default HMS on $HMS_DIR and closes it on shutdown; an injected HMS stays open.
    create_app() 在 $HMS_DIR 上构建离线默认 HMS 并在关闭时关闭它；注入的 HMS 保持打开。

    Input / 输入：
        tmp_path: pytest temporary directory (HMS_DIR points here).
        tmp_path：pytest 临时目录（HMS_DIR 指向这里）。

        monkeypatch: pytest monkeypatch (spies on HMS.close).
        monkeypatch：pytest 的 monkeypatch（监视 HMS.close）。

        test_client_cls: The TestClient class fixture.
        test_client_cls：TestClient 类夹具。

    Output / 输出：
        Assertions on app.state, the default providers, the store path and the close() calls.
        对 app.state、默认提供者、存储路径与 close() 调用的断言。

    State effect / 状态影响：
        Creates tmp_path/hms.db.
        创建 tmp_path/hms.db。
    """
    closed: list[int] = []
    original_close = HMS.close
    monkeypatch.setattr(HMS, "close", lambda self: (closed.append(id(self)), original_close(self)))
    app = create_app()
    facade = app.state.hms
    assert app.state.owns_hms is True and app.state.server_token is None
    assert facade.config.store.path == str(tmp_path / "hms.db")
    assert (facade.config.llm.provider, facade.config.embedder.provider, facade.config.vector_store.provider) == (
        "rule", "hash", "sqlite",
    )  # fmt: skip
    with test_client_cls(app) as client:
        assert client.post("/memories", json={"messages": LATTE, **SCOPE}).json()["results"][0]["memory"] == LATTE_FACT
    assert closed == [id(facade)] and (tmp_path / "hms.db").exists()
    injected = HMS({"vector_store": {"provider": "memory"}, "store": {"path": ":memory:"}})
    borrowed = create_app(injected)
    assert borrowed.state.owns_hms is False
    with test_client_cls(borrowed) as client:
        assert client.get("/health").status_code == 200
    assert closed == [id(facade)] and injected.get("mem_missing") is None
    injected.close()
    configured = create_app(config={"vector_store": {"provider": "memory"}, "store": {"path": ":memory:"}})
    assert configured.state.owns_hms is True and configured.state.hms.config.vector_store.provider == "memory"
    configured.state.hms.close()


def test_dependency_guards_and_main(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing fastapi/uvicorn → DependencyError(extra='server'); main() serves create_app(config) under uvicorn.run.
    缺少 fastapi/uvicorn → DependencyError(extra='server')；main() 通过 uvicorn.run 运行 create_app(config)。

    Input / 输入：
        tmp_path: pytest temporary directory (holds the --config JSON).
        tmp_path：pytest 临时目录（存放 --config 的 JSON）。

        monkeypatch: pytest monkeypatch (import guard, uvicorn stub).
        monkeypatch：pytest 的 monkeypatch（导入守卫、uvicorn 桩）。

    Output / 输出：
        Assertions on the DependencyError details, the uvicorn.run arguments and the pyproject console script.
        对 DependencyError 细节、uvicorn.run 参数与 pyproject 控制台脚本的断言。

    State effect / 状态影响：
        None (uvicorn is stubbed; no port is bound).
        无（uvicorn 被桩替代；不绑定端口）。
    """
    monkeypatch.setattr(server_app, "_IMPORT_ERROR", ImportError("no fastapi"))
    with pytest.raises(DependencyError) as no_fastapi:
        create_app()
    assert no_fastapi.value.details == {"package": "fastapi", "extra": "server"}
    assert no_fastapi.value.suggestion == "pip install 'hms-elite[server]'"
    monkeypatch.setattr(server_app, "_IMPORT_ERROR", None)

    def no_uvicorn() -> Any:
        """Stand in for a missing uvicorn.
        模拟缺失的 uvicorn。
        """
        raise DependencyError("no uvicorn", details={"package": "uvicorn", "extra": "server"})

    monkeypatch.setattr(server_app, "_import_uvicorn", no_uvicorn)
    with pytest.raises(DependencyError) as missing:
        main([])
    assert missing.value.details == {"package": "uvicorn", "extra": "server"}

    calls: list[tuple[Any, dict]] = []

    class _Uvicorn:
        """uvicorn stand-in recording run() calls.
        记录 run() 调用的 uvicorn 替身。
        """

        @staticmethod
        def run(app: Any, **kwargs: Any) -> None:
            """Record the call instead of binding a port.
            记录调用而不绑定端口。
            """
            calls.append((app, kwargs))

    monkeypatch.setattr(server_app, "_import_uvicorn", lambda: _Uvicorn)
    config_path = tmp_path / "hms.json"
    config_path.write_text(
        json.dumps({"vector_store": {"provider": "memory"}, "store": {"path": ":memory:"}}), encoding="utf-8"
    )
    argv = ["--host", "0.0.0.0", "--port", "9", "--config", str(config_path), "--log-level", "warning"]
    assert main(argv) == 0 and len(calls) == 1
    app, kwargs = calls[0]
    assert kwargs == {"host": "0.0.0.0", "port": 9, "log_level": "warning"}
    assert app.state.owns_hms is True and app.state.hms.config.vector_store.provider == "memory"
    app.state.hms.close()
    assert main([]) == 0 and calls[1][1] == {"host": "127.0.0.1", "port": 8000, "log_level": "info"}
    calls[1][0].state.hms.close()
    assert 'hms-server = "hms_elite.server.app:main"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_import_is_side_effect_free(tmp_path: pathlib.Path) -> None:
    """Importing hms_elite.server in a fresh interpreter creates nothing under $HMS_DIR or $HOME.
    在全新解释器中导入 hms_elite.server 不会在 $HMS_DIR 或 $HOME 下创建任何文件。

    Input / 输入：
        tmp_path: pytest temporary directory used as both HMS_DIR and HOME of the child process.
        tmp_path：作为子进程 HMS_DIR 与 HOME 的 pytest 临时目录。

    Output / 输出：
        Assertions on the child's exit status and on tmp_path staying empty.
        对子进程退出状态与 tmp_path 保持为空的断言。

    State effect / 状态影响：
        Spawns one Python child process.
        启动一个 Python 子进程。
    """
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HMS_DIR": str(tmp_path), "HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"}
    code = "import hms_elite.server, hms_elite.server.app, hms_elite.server.deps, hms_elite.server.schemas; print('ok')"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr
    assert sorted(tmp_path.iterdir()) == [home] and list(home.iterdir()) == []


def test_merge_scope_precedence() -> None:
    """merge_scope: explicit non-empty ids override filter keys; empty ids are ignored; None filters → {}.
    merge_scope：非空的显式 id 覆盖过滤键；空 id 被忽略；filters 为 None → {}。

    Input / 输入：
        none.
        无。

    Output / 输出：
        Assertions on the merged dicts.
        对合并后字典的断言。

    State effect / 状态影响：
        None.
        无。
    """
    assert merge_scope(None) == {}
    assert merge_scope({"user_id": "x", "topic": "t"}, user_id=U, run_id="") == {"user_id": U, "topic": "t"}
    original = {"agent_id": "coder"}
    assert merge_scope(original, run_id=RUN) == {"agent_id": "coder", "run_id": RUN}
    assert original == {"agent_id": "coder"}
