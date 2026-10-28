"""Real-LLM arms for EgoLife: official mem0ai 2.0.20 (milvus-lite + fastembed + deepseek) versus HMS Elite (deepseek).
EgoLife 的真实 LLM 臂：官方 mem0ai 2.0.20（milvus-lite + fastembed + deepseek）对 HMS Elite（deepseek）。
"""

# Purpose (EN): One DeepSeek client (timeout 60 s, 2 retries with exponential backoff, every attempt recorded in
#               accounting.Meter) is spliced into both libraries' provider objects; the arms expose retain_batch /
#               recall / (card) / (hybrid, EliteRealArm only: hms.context()) / stats / close; answer(order=
#               'question_first' | 'memories_first') renders their build_question_prompt verbatim plus a "Memories:"
#               block (after or before the question) and parses the single letter with their parse_choice.
# 目的（中）：一个 DeepSeek 客户端（超时 60 s、指数退避重试 2 次、每次尝试记入 accounting.Meter）被接入两个库的提供者
#            对象；各臂暴露 retain_batch / recall /（card）/（hybrid，仅 EliteRealArm：hms.context()）/ stats / close；
#            answer(order='question_first' | 'memories_first') 逐字用对方的 build_question_prompt 拼 "Memories:" 段
#            （放在问题之后或之前），并用对方的 parse_choice 解析唯一字母。

# Input (EN): Batches of Window objects (<= 25 per add), question items, query instants, the .env key file path.
# 输入（中）：Window 批次（每次 add ≤ 25 个）、题目、查询时刻、.env 密钥文件路径。

# Output (EN): Recall rows {batch_id, text, score, memory_id}, card dicts, AnswerResult, arm stats.
# 输出（中）：召回行 {batch_id, text, score, memory_id}、卡片字典、AnswerResult、臂统计。

# State effect (EN): Sets DEEPSEEK_API_KEY in this process only (never printed or written); MEM0_TELEMETRY is forced
#                    off before mem0 is imported; each arm writes only inside its own directory.
# 状态影响（中）：仅在本进程设置 DEEPSEEK_API_KEY（绝不打印或写出）；导入 mem0 之前强制关闭 MEM0_TELEMETRY；每个臂只写
#                自己的目录。

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ["MEM0_TELEMETRY"] = "False"  # must precede `import mem0` / 必须早于 `import mem0`

import openai  # noqa: E402
from accounting import MODEL, ApiCall, Meter, rate_tier, utc_now  # noqa: E402
from loader import Question  # noqa: E402
from mem0.llms.deepseek import DeepSeekLLM as Mem0DeepSeekLLM  # noqa: E402
from mem0.memory.main import Memory  # noqa: E402
from mem0.vector_stores.base import VectorStoreBase  # noqa: E402
from windows import Window  # noqa: E402

from hms_elite import HMS, HMSConfig  # noqa: E402
from hms_elite.ingest.lines import NO_SECTION_MARK  # noqa: E402
from hms_elite.llms.deepseek import DeepSeekLLM as EliteDeepSeekLLM  # noqa: E402

BANK = "egolife_a1_jake_real"
TOP_K = 20
THRESHOLD = 0.1
BATCH_SIZE = 25
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMS = 384


def embedder_identity(model: str, dims: int) -> str:
    """The stats() embedder string: 'fastembed:<model>' for the default (model, dims) — exactly what the recorded
    English runs carry — and 'fastembed:<model>@<dims>' for any other embedder (EgoLife-ZH: bge-small-zh-v1.5@512).
    stats() 的嵌入器字符串：默认 (model, dims) 为 'fastembed:<model>'——与已记录的英文运行完全一致——其它嵌入器为
    'fastembed:<model>@<dims>'（EgoLife-ZH：bge-small-zh-v1.5@512）。
    """
    if model == EMBED_MODEL and int(dims) == EMBEDDING_DIMS:
        return f"fastembed:{model}"
    return f"fastembed:{model}@{dims}"
BASE_URL = "https://api.deepseek.com"
TIMEOUT_S = 60.0
MAX_RETRIES = 2
BACKOFF_S = (1.0, 2.0)  # sleep before retry 1, retry 2 / 第 1、2 次重试前的等待
RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
EXTRACT_TEMPERATURE = 0.0
EXTRACT_MAX_TOKENS = 8192  # both libraries default to 2000; raised identically so 25-window JSON is not truncated / 两库默认 2000，同样提高以免 25 窗口的 JSON 被截断
ANSWER_MAX_TOKENS = 8
ANSWER_TEMPERATURE = 0.0
# deepseek-flash reasons by default (probe 2026-09-10: max_tokens=8 -> content '' , reasoning_tokens=8, finish=length;
# with thinking disabled -> content 'B', 1 completion token). Disabled for every call of every arm and stage.
# deepseek-flash 默认开启思考（2026-09-10 探针：max_tokens=8 -> content ''、reasoning_tokens=8、finish=length；
# 关闭思考 -> content 'B'、1 个补全 token）。对所有臂、所有阶段的每次调用统一关闭。
THINKING = {"type": "disabled"}
ENGINE_NAMES = ("mem0_real", "elite")
LABELS = ("mem0_real", "elite_rows", "elite_card")
THEIR_PROTOCOL = Path("/Users/lazy/Desktop/hms_lite-main/benchmarks/egolife.py")

# ---------------------------------------------------------------------------
# Copied verbatim from /Users/lazy/Desktop/hms_lite-main/benchmarks/egolife.py (their protocol implementation).
# 逐字复制自 /Users/lazy/Desktop/hms_lite-main/benchmarks/egolife.py（对方的协议实现）。
# ---------------------------------------------------------------------------
_CHOICE = re.compile(r"(?<![A-Za-z])[A-D](?![A-Za-z])")


def build_question_prompt(item: Mapping[str, Any]) -> str:
    """Render only public English question and choices.
    只渲染公开的英文题干与选项。

    Input / 输入：
        item: Raw QA item with 'question' and 'choice_a'..'choice_d'.
        item：含 'question' 与 'choice_a'..'choice_d' 的原始 QA 条目。

    Output / 输出：
        The prompt text (their wording, verbatim).
        提示词文本（对方措辞，逐字）。

    State effect / 状态影响：
        None.
        无。
    """
    question = item.get("question")
    choices = [item.get(f"choice_{letter}") for letter in "abcd"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("English question is required")
    if not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise ValueError("All four English choices are required")
    return "\n".join(
        [
            "Answer the multiple-choice question using only the supplied memory evidence.",
            "Return exactly one uppercase letter: A, B, C, or D.",
            "",
            f"Question: {question.strip()}",
            *(f"{letter.upper()}. {str(choice).strip()}" for letter, choice in zip("abcd", choices, strict=True)),
        ]
    )


def parse_choice(text: str) -> str | None:
    """Return an answer only when one distinct uppercase choice is present.
    仅当恰有一个不同的大写选项字母时返回答案。

    Input / 输入：
        text: Model reply.
        text：模型回复。

    Output / 输出：
        'A'..'D' or None.
        'A'..'D' 或 None。

    State effect / 状态影响：
        None.
        无。
    """
    matches = set(_CHOICE.findall(text))
    return next(iter(matches)) if len(matches) == 1 else None


def their_protocol_digest() -> dict[str, Any]:
    """sha256 of their egolife.py and of the two copied function bodies, when the Desktop copy is present.
    对方 egolife.py 及两个被复制函数体的 sha256（桌面副本存在时）。

    Input / 输入：
        none.
        无。

    Output / 输出：
        {'path', 'present', 'file_sha256', 'functions_identical'}.
        {'path', 'present', 'file_sha256', 'functions_identical'}。

    State effect / 状态影响：
        None.
        无。
    """
    out: dict[str, Any] = {"path": str(THEIR_PROTOCOL), "present": THEIR_PROTOCOL.exists()}
    if not THEIR_PROTOCOL.exists():
        return out
    src = THEIR_PROTOCOL.read_text(encoding="utf-8")
    out["file_sha256"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
    prompt_body = src[src.index("def build_question_prompt") :]
    prompt_body = prompt_body[prompt_body.index('return "\\n".join(') : prompt_body.index("\n\n\ndef ")]
    parse_body = src[src.index("def parse_choice") :]
    parse_body = parse_body[parse_body.index("matches = ") : parse_body.index("\n\n\ndef ")]
    out["functions_identical"] = (
        'return "\\n".join(\n        [\n            "Answer the multiple-choice question using only the supplied memory evidence."'
        in prompt_body
        and "next(iter(matches)) if len(matches) == 1 else None" in parse_body
    )
    return out


def load_api_key(env_path: Path) -> None:
    """Read DEEPSEEK_API_KEY from a one-line .env file into os.environ (value never returned, printed or logged).
    从单行 .env 文件读取 DEEPSEEK_API_KEY 到 os.environ（值绝不返回、打印或记录）。

    Input / 输入：
        env_path: The .env file.
        env_path：.env 文件。

    Output / 输出：
        None; SystemExit when the variable is absent.
        无；变量缺失时抛 SystemExit。

    State effect / 状态影响：
        Sets os.environ['DEEPSEEK_API_KEY'] for this process.
        为本进程设置 os.environ['DEEPSEEK_API_KEY']。
    """
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DEEPSEEK_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                os.environ["DEEPSEEK_API_KEY"] = value
                return
    raise SystemExit(f"DEEPSEEK_API_KEY not found in {env_path}")


@dataclass
class Batch:
    """One add() call: up to BATCH_SIZE windows released at one timeline step.
    一次 add() 调用：同一时间线步骤释放的至多 BATCH_SIZE 个窗口。
    """

    batch_id: str
    position: int  # timeline step / 时间线步骤
    windows: list[Window]

    @property
    def window_ids(self) -> list[str]:
        """Ids of the windows in this batch.
        本批次窗口的 id。

        Input / 输入：
            none.
            无。

        Output / 输出：
            List of window ids in order.
            按序的窗口 id 列表。

        State effect / 状态影响：
            None.
            无。
        """
        return [w.window_id for w in self.windows]

    def messages(self) -> list[dict[str, str]]:
        """One user message per window (same shape for both libraries).
        每个窗口一条 user 消息（两库同形）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            [{'role': 'user', 'content': window.text}, ...].
            [{'role': 'user', 'content': 窗口文本}, ...]。

        State effect / 状态影响：
            None.
            无。
        """
        return [{"role": "user", "content": w.text} for w in self.windows]

    def metadata(self) -> dict[str, Any]:
        """Scalar-only metadata attached to every memory the batch produces (window list lives in batches.json).
        附加到本批次产出的每条记忆上的纯标量元数据（窗口列表放在 batches.json）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            {'batch_id', 'position', 'day', 'start', 'end', 'window_from', 'window_to', 'windows'}.
            {'batch_id', 'position', 'day', 'start', 'end', 'window_from', 'window_to', 'windows'}。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "batch_id": self.batch_id,
            "position": self.position,
            "day": self.windows[0].day,
            "start": self.windows[0].start_cs,
            "end": max(w.end_cs for w in self.windows),
            "window_from": self.windows[0].window_id,
            "window_to": self.windows[-1].window_id,
            "windows": len(self.windows),
        }


def plan_batches(steps: list[Any], size: int = BATCH_SIZE) -> list[Batch]:
    """Chunk each step's new_windows into batches of <= size (a batch never spans two steps).
    把每个步骤的 new_windows 切成 ≤ size 的批次（批次不跨步骤）。

    Input / 输入：
        steps: timeline.Step list.
        steps：timeline.Step 列表。

        size: Max windows per add().
        size：每次 add() 的最大窗口数。

    Output / 输出：
        Batches in stream order with ids b0000, b0001, ...
        按流顺序的批次，id 为 b0000、b0001、...

    State effect / 状态影响：
        None.
        无。
    """
    out: list[Batch] = []
    for step in steps:
        ws = list(step.new_windows)
        for i in range(0, len(ws), size):
            out.append(Batch(batch_id=f"b{len(out):04d}", position=step.position, windows=ws[i : i + size]))
    return out


class DeepSeekClient:
    """OpenAI-SDK client for api.deepseek.com with per-attempt accounting; exposes `.chat.completions.create`.
    面向 api.deepseek.com 的 OpenAI SDK 客户端，逐次尝试记账；暴露 `.chat.completions.create`。
    """

    def __init__(self, meter: Meter, *, model: str = MODEL, timeout_s: float = TIMEOUT_S, max_retries: int = MAX_RETRIES) -> None:
        """Build the SDK client with retries disabled (this class retries and records) and a fixed timeout.
        构建 SDK 客户端：关闭 SDK 自带重试（由本类重试并记录），固定超时。

        Input / 输入：
            meter: The ledger.
            meter：台账。

            model: Model name.
            model：模型名。

            timeout_s: Per-attempt timeout.
            timeout_s：每次尝试的超时。

            max_retries: Retries after the first attempt.
            max_retries：首次之后的重试次数。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Creates the SDK client from os.environ['DEEPSEEK_API_KEY'].
            用 os.environ['DEEPSEEK_API_KEY'] 创建 SDK 客户端。
        """
        self.meter = meter
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.sdk = openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL, timeout=timeout_s, max_retries=0)
        self.labels: dict[str, Any] = {"arm": "unlabelled", "stage": "extract", "question_id": None, "batch_id": None, "split": None}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.last_calls: list[ApiCall] = []
        # Optional sink for raw model output (run_real writes extract outputs to extract_raw_<engine>.jsonl).
        # 可选的原始模型输出接收器（run_real 把抽取输出写到 extract_raw_<engine>.jsonl）。
        self.raw_sink: Any = None

    @contextmanager
    def label(self, **labels: Any) -> Iterator[None]:
        """Set arm / stage / question_id / batch_id for the calls made inside the block.
        为块内的调用设置 arm / stage / question_id / batch_id。

        Input / 输入：
            labels: Any of arm, stage, question_id, batch_id.
            labels：arm、stage、question_id、batch_id 任意几项。

        Output / 输出：
            Context manager.
            上下文管理器。

        State effect / 状态影响：
            Restores the previous labels on exit.
            退出时恢复之前的标签。
        """
        previous = dict(self.labels)
        self.labels.update(labels)
        try:
            yield
        finally:
            self.labels = previous

    @staticmethod
    def _classify(exc: BaseException) -> tuple[str, int | None, bool]:
        """(status, http_status, retryable) for an SDK exception.
        SDK 异常对应的 (状态, HTTP 状态码, 是否可重试)。
        """
        if isinstance(exc, openai.APITimeoutError):
            return "timeout", None, True
        if isinstance(exc, openai.APIStatusError):
            code = int(getattr(exc, "status_code", 0) or 0)
            return "http_error", code, code in RETRY_STATUSES
        if isinstance(exc, openai.APIConnectionError):
            return "http_error", None, True
        return "http_error", None, False

    def create(self, **params: Any) -> Any:
        """chat.completions.create with up to MAX_RETRIES retries; each attempt becomes one ApiCall.
        带至多 MAX_RETRIES 次重试的 chat.completions.create；每次尝试成为一条 ApiCall。

        Input / 输入：
            params: SDK kwargs (model is forced to self.model; stream is refused).
            params：SDK 参数（model 强制为 self.model；不接受 stream）。

        Output / 输出：
            The SDK response of the successful attempt; the last exception is re-raised when every attempt fails.
            成功尝试的 SDK 响应；全部失败时重新抛出最后一个异常。

        State effect / 状态影响：
            Appends ApiCalls to the meter and to self.last_calls; sleeps BACKOFF_S between attempts.
            向台账与 self.last_calls 追加 ApiCall；尝试之间按 BACKOFF_S 等待。
        """
        params = dict(params)
        params["model"] = self.model
        params.pop("stream", None)
        extra_body = dict(params.get("extra_body") or {})
        extra_body["thinking"] = dict(THINKING)
        params["extra_body"] = extra_body
        self.last_calls = []
        arm = str(self.labels["arm"])
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
            started = utc_now()
            cold = self.meter.next_is_cold(arm)
            t0 = time.perf_counter()
            call = ApiCall(
                arm=arm,
                stage=str(self.labels["stage"]),
                question_id=self.labels.get("question_id"),
                attempt=attempt,
                started_utc=started.isoformat(),
                latency_ms=0.0,
                status="ok",
                rate_tier=rate_tier(started),
                model=self.model,
                batch_id=self.labels.get("batch_id"),
                cold_start=cold,
                split=self.labels.get("split"),
            )
            try:
                response = self.sdk.chat.completions.create(**params)
            except Exception as exc:  # noqa: BLE001 - every failure is a ledger row
                call.latency_ms = (time.perf_counter() - t0) * 1000.0
                call.status, call.http_status, retryable = self._classify(exc)
                call.error = f"{type(exc).__name__}: {str(exc)[:300]}"
                self.meter.add(call)
                self.last_calls.append(call)
                last_exc = exc
                if not retryable:
                    break
                continue
            call.latency_ms = (time.perf_counter() - t0) * 1000.0
            usage = getattr(response, "usage", None)
            if usage is not None:
                dump = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
                call.prompt_tokens = _int_or_none(dump.get("prompt_tokens"))
                call.completion_tokens = _int_or_none(dump.get("completion_tokens"))
                call.cache_hit_tokens = _int_or_none(dump.get("prompt_cache_hit_tokens"))
                call.cache_miss_tokens = _int_or_none(dump.get("prompt_cache_miss_tokens"))
                details = dump.get("completion_tokens_details") or {}
                call.reasoning_tokens = _int_or_none(details.get("reasoning_tokens")) if isinstance(details, dict) else None
            choice = response.choices[0] if getattr(response, "choices", None) else None
            call.finish_reason = getattr(choice, "finish_reason", None) if choice else None
            message = getattr(choice, "message", None) if choice else None
            content = getattr(message, "content", None)
            call.content_chars = len(content) if isinstance(content, str) else None
            if not (isinstance(content, str) and content.strip()):
                call.status = "empty"
                call.error = "empty_content"
            self.meter.add(call)
            self.last_calls.append(call)
            if self.raw_sink is not None and call.stage == "extract":
                reasoning = getattr(message, "reasoning_content", None)
                self.raw_sink(
                    {
                        "arm": call.arm, "stage": call.stage, "batch_id": call.batch_id, "split": call.split, "attempt": attempt,
                        "started_utc": call.started_utc, "finish_reason": call.finish_reason,
                        "completion_tokens": call.completion_tokens, "reasoning_tokens": call.reasoning_tokens,
                        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                        "content": content if isinstance(content, str) else None,
                    }
                )
            return response
        assert last_exc is not None
        raise last_exc

    def mark_last(self, *, status: str, error: str | None) -> None:
        """Overwrite status / error of the most recent attempt (answer parsing marks invalid output as 'empty').
        改写最近一次尝试的状态 / 错误（答题解析把无效输出标为 'empty'）。

        Input / 输入：
            status: New status.
            status：新状态。

            error: New error text.
            error：新错误文本。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Mutates the last ApiCall in place.
            原地修改最后一条 ApiCall。
        """
        if self.last_calls:
            self.last_calls[-1].status = status
            self.last_calls[-1].error = error


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def instrument_embedder(obj: Any, meter: Meter, arm: str, client: DeepSeekClient) -> None:
    """Wrap obj.embed / obj.embed_batch on the instance: count texts and wall time into the meter (stage from labels).
    在实例上包装 obj.embed / obj.embed_batch：把文本数与墙钟耗时计入台账（阶段取自标签）。

    Input / 输入：
        obj: mem0 EmbeddingBase or hms_elite EmbedderBase instance.
        obj：mem0 EmbeddingBase 或 hms_elite EmbedderBase 实例。

        meter: The ledger.
        meter：台账。

        arm: Arm label.
        arm：臂标签。

        client: Whose labels give the current stage.
        client：其标签提供当前阶段。

    Output / 输出：
        None.
        无。

    State effect / 状态影响：
        Replaces the two bound methods on the instance; nested embed calls inside embed_batch are not double counted.
        替换实例上的两个绑定方法；embed_batch 内部的 embed 调用不重复计数。
    """
    state = {"depth": 0}
    original_embed = obj.embed
    original_batch = getattr(obj, "embed_batch", None)

    def _record(n: int, t0: float) -> None:
        stage = str(client.labels.get("stage") or "other")
        meter.add_embedding(arm, stage if stage in ("extract", "recall") else "other", n, (time.perf_counter() - t0) * 1000.0)

    def embed(text: Any, *args: Any, **kwargs: Any) -> Any:
        if state["depth"]:
            return original_embed(text, *args, **kwargs)
        state["depth"] += 1
        t0 = time.perf_counter()
        try:
            return original_embed(text, *args, **kwargs)
        finally:
            state["depth"] -= 1
            _record(1, t0)

    def embed_batch(texts: Any, *args: Any, **kwargs: Any) -> Any:
        texts = list(texts)
        if state["depth"]:
            return original_batch(texts, *args, **kwargs)
        state["depth"] += 1
        t0 = time.perf_counter()
        try:
            return original_batch(texts, *args, **kwargs)
        finally:
            state["depth"] -= 1
            _record(len(texts), t0)

    obj.embed = embed
    if callable(original_batch):
        obj.embed_batch = embed_batch


@dataclass
class AnswerResult:
    """Outcome of one answer() call.
    一次 answer() 调用的结果。
    """

    letter: str | None
    raw: str | None
    status: str  # ok | empty | timeout | http_error / 状态
    prompt_chars: int
    calls: list[ApiCall] = field(default_factory=list)
    model_ms: float = 0.0  # sum of attempt latencies / 各尝试延迟之和
    local_ms: float = 0.0
    wall_ms: float = 0.0


def build_prompt(question: Question, context: str, order: str) -> str:
    """The single user message: 'question_first' = their question prompt, then 'Memories:\\n<context>' (the shape of every run before 2026-09-18); 'memories_first' = 'Memories:\\n<context>\\n\\n<question prompt>' so consecutive questions sharing a context prefix (HMS.context()'s session card) share the prompt prefix. Blank context renders as '(none)' in both orders.
    唯一的 user 消息：'question_first' = 对方题目提示词在前，再 'Memories:\\n<上下文>'（2026-09-18 之前每次运行的形状）；'memories_first' = 'Memories:\\n<上下文>\\n\\n<题目提示词>'，使共享上下文前缀（HMS.context() 的会话卡片）的连续题目共享提示词前缀。两种顺序下空白上下文都渲染为 '(none)'。

    Input / 输入：
        question: The question (public fields only are rendered).
        question：题目（只渲染公开字段）。

        context: Evidence block body.
        context：证据段正文。

        order: 'question_first' | 'memories_first'.
        order：'question_first' | 'memories_first'。

    Output / 输出：
        The prompt text.
        提示词文本。

    State effect / 状态影响：
        None.
        无。
    """
    body = context if context.strip() else "(none)"
    head = build_question_prompt(question.raw)
    if order == "memories_first":
        return "Memories:\n" + body + "\n\n" + head
    if order != "question_first":
        raise ValueError(f"unknown prompt order {order!r}")
    return head + "\n\nMemories:\n" + body


def answer(
    client: DeepSeekClient, question: Question, context_text: str, *, arm: str, order: str = "question_first"
) -> AnswerResult:
    """Ask deepseek-flash once (temperature 0, max_tokens 8) with build_prompt(question, context, order); parse one letter.
    以 build_prompt(question, context, order) 向 deepseek-flash 提问一次（temperature 0、max_tokens 8）；解析唯一字母。

    Input / 输入：
        client: The accounting client.
        client：记账客户端。

        question: Question (only .raw public fields are rendered).
        question：题目（只渲染 .raw 中的公开字段）。

        context_text: The evidence block body ('' -> '(none)').
        context_text：证据段正文（'' -> '(none)'）。

        arm: Label for the ledger (mem0_real | elite_rows | elite_hybrid | elite_card).
        arm：台账标签（mem0_real | elite_rows | elite_hybrid | elite_card）。

        order: 'question_first' (default; rows / card contexts) | 'memories_first' (the hybrid context, whose session-card prefix is stable across questions).
        order：'question_first'（默认；记忆行 / 卡片上下文）| 'memories_first'（混合上下文，其会话卡片前缀在题目之间稳定）。

    Output / 输出：
        AnswerResult; an empty / unparsable reply is status 'empty' and letter None.
        AnswerResult；空回复 / 无法解析时 status 为 'empty'，letter 为 None。

    State effect / 状态影响：
        One to three ApiCalls appended to the meter.
        向台账追加 1 到 3 条 ApiCall。
    """
    t_wall = time.perf_counter()
    prompt = build_prompt(question, context_text, order)
    with client.label(arm=arm, stage="answer", question_id=question.id, batch_id=None):
        try:
            response = client.create(
                messages=[{"role": "user", "content": prompt}], temperature=ANSWER_TEMPERATURE, max_tokens=ANSWER_MAX_TOKENS
            )
        except Exception:  # noqa: BLE001 - recorded in the ledger by create()
            calls = list(client.last_calls)
            return AnswerResult(
                letter=None,
                raw=None,
                status=calls[-1].status if calls else "http_error",
                prompt_chars=len(prompt),
                calls=calls,
                model_ms=sum(c.latency_ms for c in calls),
                local_ms=0.0,
                wall_ms=(time.perf_counter() - t_wall) * 1000.0,
            )
    t_local = time.perf_counter()
    content = response.choices[0].message.content if response.choices else None
    raw = content if isinstance(content, str) else None
    letter = parse_choice(raw) if raw else None
    if letter is None:
        client.mark_last(status="empty", error="invalid_output" if (raw and raw.strip()) else "empty_content")
    calls = list(client.last_calls)
    return AnswerResult(
        letter=letter,
        raw=raw,
        status=calls[-1].status if calls else "ok",
        prompt_chars=len(prompt),
        calls=calls,
        model_ms=sum(c.latency_ms for c in calls),
        local_ms=(time.perf_counter() - t_local) * 1000.0,
        wall_ms=(time.perf_counter() - t_wall) * 1000.0,
    )


def rows_to_context(rows: list[dict[str, Any]]) -> str:
    """'- text' lines in engine rank order (the mem0 'organize' shape).
    按引擎排名的 '- 文本' 行（mem0 organize 的形状）。

    Input / 输入：
        rows: Recall rows.
        rows：召回行。

    Output / 输出：
        Joined text ('' for no rows).
        连接后的文本（无行时为 ''）。

    State effect / 状态影响：
        None.
        无。
    """
    return "\n".join(f"- {str(r.get('text') or '').strip()}" for r in rows if str(r.get("text") or "").strip())


class LogCapture(logging.Handler):
    """Collect WARNING+ records of the 'mem0' and 'hms_elite' loggers while a block runs (silent parse failures).
    收集块运行期间 'mem0' 与 'hms_elite' 日志器的 WARNING+ 记录（静默的解析失败）。
    """

    LOGGERS = ("mem0", "hms_elite")

    def __init__(self) -> None:
        """Start with an empty message list at WARNING level.
        以 WARNING 级别与空消息列表开始。
        """
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store `logger:LEVEL: message` (first 300 chars).
        存储 `日志器:级别: 消息`（前 300 字符）。

        Input / 输入：
            record: The log record.
            record：日志记录。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Appends to messages.
            追加到 messages。
        """
        try:
            self.messages.append(f"{record.name}:{record.levelname}: {record.getMessage()[:300]}")
        except Exception:  # pragma: no cover - never let logging break the run
            self.messages.append(f"{record.name}:{record.levelname}: <unformattable>")

    def __enter__(self) -> LogCapture:
        for name in self.LOGGERS:
            logging.getLogger(name).addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        for name in self.LOGGERS:
            logging.getLogger(name).removeHandler(self)


def _path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())



TRUNCATED = "length"  # finish_reason of an output cut at max_tokens / 输出在 max_tokens 处被截断的 finish_reason
# Prefix of a LogCapture message written by the lines parser (`<logger>:<LEVEL>: ...`).
# 行式解析器写入 LogCapture 的消息前缀（`<日志器>:<级别>: ...`）。
LINES_LOGGER = "hms_elite.ingest.lines:"


def sub_metadata(batch: Batch, windows: list[Window], split: str | None) -> dict[str, Any]:
    """batch.metadata() re-scoped to a window slice (window_from / window_to / windows / start / end / split).
    把 batch.metadata() 收窄到一个窗口切片（window_from / window_to / windows / start / end / split）。

    Input / 输入：
        batch: The batch (batch_id / position / day are kept).
        batch：批次（保留 batch_id / position / day）。

        windows: The slice.
        windows：切片。

        split: '<lo>-<hi>' or None for the whole batch.
        split：'<lo>-<hi>'，整批为 None。

    Output / 输出：
        Scalar-only metadata dict.
        纯标量元数据字典。

    State effect / 状态影响：
        None.
        无。
    """
    meta = batch.metadata()
    meta.update(
        {
            "start": windows[0].start_cs,
            "end": max(w.end_cs for w in windows),
            "window_from": windows[0].window_id,
            "window_to": windows[-1].window_id,
            "windows": len(windows),
            "split": split or "",
        }
    )
    return meta


def retain_with_bisect(arm: Any, batch: Batch) -> dict[str, Any]:
    """add() the batch; any extraction attempt with finish_reason == 'length' is rejected and its window slice is
    re-submitted as two halves (recursively, down to one window). Every sub-add is a priced request in the ledger.
    对批次执行 add()；任何 finish_reason == 'length' 的抽取尝试都被拒绝，其窗口切片拆成两半重提（递归到单窗口为止）。
    每次子 add 都是台账里的一次计费请求。

    Input / 输入：
        arm: Object with .name, .client, .meter and ._add(windows, metadata, timestamp) -> add() result dict.
        arm：具备 .name、.client、.meter 与 ._add(windows, metadata, timestamp) -> add() 结果字典的对象。

        batch: The batch.
        batch：批次。

    Output / 输出：
        {'batch_id', 'ok', 'added', 'error', 'wall_ms', 'model_ms', 'calls', 'library_warnings', 'finish_reasons',
         'subs', 'adds', 'bisected', 'truncated_attempts', 'windows', 'windows_stored', 'windows_lost', 'events',
         'downgraded' ({verdict: count} of guard-downgraded verdicts; {} for mem0)};
        subs = one record per add() in submission order (outcome ∈ ok | truncated | truncated_unresolved |
        invalid_json_retry | invalid_json_bisect | invalid_json | schema_rejected_all | empty_extraction | api_error).
        {...}；subs = 每次 add() 一条记录，按提交顺序（outcome ∈ ok | truncated | truncated_unresolved | invalid_json |
        schema_rejected_all | empty_extraction | api_error）。

    State effect / 状态影响：
        Writes the store; a truncated slice of one window, an API failure after retries, an invalid-JSON reply or a
        reply whose every item the library rejected leaves that slice's windows unstored (windows_lost > 0).
        写库；单窗口仍被截断、重试后仍失败的 API、非法 JSON 回复或全部条目被库拒绝的回复，会让该切片的窗口未入库
        （windows_lost > 0）。
    """
    t0 = time.perf_counter()
    first_call = len(arm.meter.calls)
    subs: list[dict[str, Any]] = []
    queue: list[tuple[int, int]] = [(0, len(batch.windows))]  # window offsets [lo, hi) inside the batch / 批次内窗口偏移
    ok = True
    error: str | None = None
    # A reply that is not parseable JSON (finish_reason=stop, but the library extracted nothing) is re-submitted once
    # as the same slice; if it fails again the slice is bisected like a truncated one. Every retry is a priced request.
    # 不可解析的回复（finish_reason=stop 但库什么都没抽出）同一切片重提一次；再失败则像截断一样对半拆分。每次重试都计费。
    parse_retried: set[tuple[int, int]] = set()
    while queue:
        lo, hi = queue.pop(0)
        windows = batch.windows[lo:hi]
        split = None if (lo, hi) == (0, len(batch.windows)) else f"{lo}-{hi}"
        before = len(arm.meter.calls)
        t_sub = time.perf_counter()
        result: Any = None
        with arm.client.label(arm=arm.name, stage="extract", question_id=None, batch_id=batch.batch_id, split=split), LogCapture() as logs:
            try:
                result = arm._add(windows, sub_metadata(batch, windows, split), windows[0].start_iso)
                rows = result.get("results", []) if isinstance(result, dict) else (result or [])
                added, sub_error = len(rows), None
            except Exception as exc:  # noqa: BLE001 - kept in the results
                rows, added, sub_error = [], 0, f"{type(exc).__name__}: {str(exc)[:300]}"
        calls = arm.meter.calls[before:]
        finish = [c.finish_reason for c in calls]
        truncated = any(fr == TRUNCATED for fr in finish)
        # Unusable reply: the JSON path logs "Error parsing ... response"; the lines path (CONTRACT §6.6) never raises
        # and logs NO_SECTION_MARK instead when nothing in the reply opened a section (prose, a JSON object, '').
        # Warnings of the lines parser itself (hms_elite.ingest.lines: a DEC line out of range, an EP line without
        # slots, a truncated last line) are line drops, not schema rejections: counted apart as lines_dropped.
        # 不可用的回复：JSON 路径记 "Error parsing ... response"；行式路径（CONTRACT §6.6）从不抛错，回复没有开启任何分节
        # （散文、JSON 对象、空串）时记 NO_SECTION_MARK。行式解析器自身的警告（hms_elite.ingest.lines：DEC 行越界、EP 行
        # 无槽位、末行被截断）是丢行而非模式拒绝：另计为 lines_dropped。
        parse_error = sum(1 for w in logs.messages if ("Error parsing" in w and "response" in w) or NO_SECTION_MARK in w)
        non_object = sum(1 for w in logs.messages if "reply is not a JSON object" in w)
        lines_dropped = sum(1 for w in logs.messages if w.startswith(LINES_LOGGER) and NO_SECTION_MARK not in w)
        skipped = sum(
            1 for w in logs.messages
            if "memory_extraction" in w and ("skip" in w.lower() or "malformed" in w.lower()) and not w.startswith(LINES_LOGGER)
        )
        if sub_error is not None:
            outcome = "api_error"
        elif truncated:
            outcome = "truncated" if len(windows) > 1 else "truncated_unresolved"
        elif parse_error or non_object:
            if (lo, hi) not in parse_retried:
                outcome = "invalid_json_retry"
            elif len(windows) > 1:
                outcome = "invalid_json_bisect"
            else:
                outcome = "invalid_json"
        elif added == 0:
            outcome = "schema_rejected_all" if skipped else "empty_extraction"
        else:
            outcome = "ok"
        events: dict[str, int] = {}
        for r in rows:
            ev = str((r or {}).get("event", "?")) if isinstance(r, dict) else "?"
            events[ev] = events.get(ev, 0) + 1
        # Verdicts the ingest.verdict_min_similarity guard turned into ADD (EliteRealArm._add attaches them as
        # 'downgraded': {verdict: count}; mem0's envelope has none → {}), so summarize() can count guard rejections per run.
        # 被 ingest.verdict_min_similarity 守卫改为 ADD 的裁决（EliteRealArm._add 以 'downgraded': {裁决: 数量} 附上；
        # mem0 的信封没有 → {}），使 summarize() 能按运行统计守卫拒绝数。
        downgraded = dict(result.get("downgraded") or {}) if isinstance(result, dict) else {}
        subs.append(
            {
                "split": split, "lo": lo, "hi": hi, "windows": len(windows),
                "window_from": windows[0].window_id, "window_to": windows[-1].window_id,
                "attempts": len(calls), "statuses": [c.status for c in calls], "finish_reasons": finish,
                "added": added, "events": events, "downgraded": downgraded, "error": sub_error, "outcome": outcome,
                "parse_error": parse_error, "non_object_reply": max(0, non_object - parse_error), "items_skipped": skipped,
                "items_skipped_attributed_to": sum(1 for w in logs.messages if "memory_extraction" in w and "skipped" in w and "attributed_to" in w),
                "lines_dropped": lines_dropped,
                "warnings": list(logs.messages),
                "wall_ms": (time.perf_counter() - t_sub) * 1000.0, "model_ms": sum(c.latency_ms for c in calls),
            }
        )
        if sub_error is not None:
            ok, error = False, sub_error
        if outcome == "invalid_json_retry":
            parse_retried.add((lo, hi))
            queue.insert(0, (lo, hi))
        if outcome in ("truncated", "invalid_json_bisect"):
            mid = lo + (hi - lo) // 2
            queue.insert(0, (mid, hi))
            queue.insert(0, (lo, mid))
    calls = arm.meter.calls[first_call:]
    leaves = [s for s in subs if s["outcome"] not in ("truncated", "invalid_json_retry", "invalid_json_bisect")]
    stored = sum(s["windows"] for s in leaves if s["added"] > 0)
    events_total: dict[str, int] = {}
    downgraded_total: dict[str, int] = {}
    for s in subs:
        for k, v in s["events"].items():
            events_total[k] = events_total.get(k, 0) + v
        for k, v in s["downgraded"].items():
            downgraded_total[k] = downgraded_total.get(k, 0) + v
    return {
        "batch_id": batch.batch_id,
        "ok": ok,
        "added": sum(s["added"] for s in subs),
        "error": error,
        "wall_ms": (time.perf_counter() - t0) * 1000.0,
        "model_ms": sum(c.latency_ms for c in calls),
        "calls": [c.to_dict() for c in calls],
        "library_warnings": [w for s in subs for w in s["warnings"]],
        "finish_reasons": [fr for s in subs for fr in s["finish_reasons"]],
        "subs": subs,
        "adds": len(subs),
        "bisected": any(s["outcome"] in ("truncated", "invalid_json_bisect") for s in subs),
        "parse_retries": sum(1 for s in subs if s["outcome"] == "invalid_json_retry"),
        "parse_bisects": sum(1 for s in subs if s["outcome"] == "invalid_json_bisect"),
        "truncated_attempts": sum(1 for fr in subs for x in fr["finish_reasons"] if x == TRUNCATED),
        "truncated_parsed_added": sum(s["added"] for s in subs if s["outcome"] == "truncated"),
        "windows": len(batch.windows),
        "windows_stored": stored,
        "windows_lost": len(batch.windows) - stored,
        "events": events_total,
        "downgraded": downgraded_total,
    }


class Mem0RealArm:
    """Official mem0ai Memory: milvus-lite (COSINE + Milvus BM25 hybrid), fastembed bge-small, deepseek-flash, infer=True.
    官方 mem0ai Memory：milvus-lite（COSINE + Milvus BM25 混合）、fastembed bge-small、deepseek-flash、infer=True。
    """

    name = "mem0_real"

    def __init__(
        self,
        workdir: Path,
        client: DeepSeekClient,
        meter: Meter,
        *,
        embed_model: str = EMBED_MODEL,
        embed_dims: int = EMBEDDING_DIMS,
    ) -> None:
        """Memory.from_config(...) then splice the accounting client into memory.llm and wrap the embedder.
        Memory.from_config(...) 后把记账客户端接入 memory.llm，并包装嵌入器。

        Input / 输入：
            workdir: Holds milvus.db and history.db.
            workdir：存放 milvus.db 与 history.db。

            client: Shared DeepSeekClient.
            client：共享 DeepSeekClient。

            meter: Shared ledger.
            meter：共享台账。

            embed_model: fastembed model name (default EMBED_MODEL); mem0's provider takes only the name, so the cache directory comes from the FASTEMBED_CACHE_PATH environment variable (run_real exports it).
            embed_model：fastembed 模型名（默认 EMBED_MODEL）；mem0 的提供者只接受模型名，缓存目录来自环境变量 FASTEMBED_CACHE_PATH（run_real 会导出）。

            embed_dims: The model's vector size (default EMBEDDING_DIMS); milvus creates the collection with it, so it must match the model.
            embed_dims：模型向量维数（默认 EMBEDDING_DIMS）；milvus 用它建集合，必须与模型一致。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Creates the store files; loads the fastembed model.
            创建库文件；加载 fastembed 模型。
        """
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.meter = meter
        self.embed_model = embed_model
        self.embed_dims = int(embed_dims)
        self.milvus_db = workdir / "milvus.db"
        self.history_db = workdir / "history.db"
        self.memory = Memory.from_config(
            {
                "vector_store": {
                    "provider": "milvus",
                    "config": {
                        "url": str(self.milvus_db),
                        "collection_name": "egolife_real",
                        "embedding_model_dims": self.embed_dims,
                        "metric_type": "COSINE",
                        "db_name": "",
                    },
                },
                "embedder": {"provider": "fastembed", "config": {"model": self.embed_model, "embedding_dims": self.embed_dims}},
                "llm": {
                    "provider": "deepseek",
                    "config": {"model": MODEL, "temperature": EXTRACT_TEMPERATURE, "max_tokens": EXTRACT_MAX_TOKENS},
                },
                "history_db_path": str(self.history_db),
            }
        )
        if not isinstance(self.memory.llm, Mem0DeepSeekLLM):
            raise RuntimeError("mem0 did not build its DeepSeek provider")
        self.memory.llm.client = client  # every extraction call now goes through the ledger / 抽取调用全部经台账
        instrument_embedder(self.memory.embedding_model, meter, self.name, client)
        store = self.memory.vector_store
        self.hybrid = bool(getattr(store, "_has_bm25_schema", False)) and (
            type(store).keyword_search is not VectorStoreBase.keyword_search
        )
        self.batches_ok = 0
        self.batches_failed = 0
        self.memories_added = 0
        self.closed = False

    def _add(self, windows: list[Window], metadata: dict[str, Any], timestamp: str) -> dict[str, Any]:
        """One add(messages=<windows>, user_id=BANK, metadata=metadata, infer=True) (mem0 has no timestamp argument).
        一次 add(messages=<windows>, user_id=BANK, metadata=metadata, infer=True)（mem0 没有 timestamp 参数）。

        Input / 输入：
            windows: Window slice.
            windows：窗口切片。

            metadata: Scalar metadata.
            metadata：标量元数据。

            timestamp: Ignored by mem0 (interface parity).
            timestamp：mem0 不使用（保持接口一致）。

        Output / 输出：
            mem0's add() result.
            mem0 的 add() 结果。

        State effect / 状态影响：
            Writes the store.
            写库。
        """
        result = self.memory.add(messages=[{"role": "user", "content": w.text} for w in windows], user_id=BANK, metadata=metadata, infer=True)
        return result if isinstance(result, dict) else {"results": list(result or [])}

    def retain_batch(self, batch: Batch) -> dict[str, Any]:
        """retain_with_bisect(self, batch): add(messages=<=25 windows, ..., infer=True), halves re-submitted on truncation.
        retain_with_bisect(self, batch)：add(messages=<=25 个窗口, ..., infer=True)，截断时两半重提。

        Input / 输入：
            batch: The batch.
            batch：批次。

        Output / 输出：
            retain_with_bisect's record.
            retain_with_bisect 的记录。

        State effect / 状态影响：
            Writes the store; counters batches_ok / batches_failed / memories_added updated.
            写库；更新 batches_ok / batches_failed / memories_added 计数。
        """
        row = retain_with_bisect(self, batch)
        if row["ok"]:
            self.batches_ok += 1
        else:
            self.batches_failed += 1
        self.memories_added += row["added"]
        return row

    def recall(self, question: str, query_iso: str) -> list[dict[str, Any]]:
        """search(question, filters={user_id}, top_k=20, threshold=0.1) -> rows in mem0 order.
        search(question, filters={user_id}, top_k=20, threshold=0.1) -> 按 mem0 顺序的行。

        Input / 输入：
            question: Public question text.
            question：公开题干。

            query_iso: Ignored by mem0 (kept for interface parity).
            query_iso：mem0 不使用（保持接口一致）。

        Output / 输出：
            [{'batch_id', 'text', 'score', 'memory_id'}].
            [{'batch_id', 'text', 'score', 'memory_id'}]。

        State effect / 状态影响：
            One local query embedding.
            一次本地查询嵌入。
        """
        with self.client.label(arm=self.name, stage="recall", question_id=None, batch_id=None):
            result = self.memory.search(question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD)
        rows = []
        for row in result.get("results", []):
            meta = row.get("metadata") or {}
            rows.append(
                {
                    "batch_id": meta.get("batch_id"),
                    "text": row.get("memory", ""),
                    "score": float(row.get("score", 0.0)),
                    "memory_id": row.get("id"),
                }
            )
        return rows

    def stats(self) -> dict[str, Any]:
        """Counts and sizes.
        计数与尺寸。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with stored_count, db_bytes, hybrid, batches, memories_added and the LLM / embedder identities.
            含 stored_count、db_bytes、hybrid、批次数、memories_added 与 LLM / 嵌入器身份的字典。

        State effect / 状态影响：
            None.
            无。
        """
        files = {"milvus.db": _path_bytes(self.milvus_db), "history.db": _path_bytes(self.history_db)}
        return {
            "engine": "mem0ai 2.0.20 Memory",
            "vector_store": "milvus (milvus-lite local file)",
            "hybrid": self.hybrid,
            "hybrid_note": "mem0's keyword channel is Milvus's built-in BM25 Function on the text field (not a fastembed sparse model)",
            "llm": f"deepseek:{MODEL}",
            # The recorded English runs (results/real/mem0_full204) carry 'fastembed:<model>'; the '@<dims>' suffix is
            # appended only for a non-default embedder so the default stats string stays byte-identical.
            # 已记录的英文运行（results/real/mem0_full204）写的是 'fastembed:<model>'；'@<dims>' 后缀只在非默认嵌入器时追加，
            # 使默认的 stats 字符串逐字节不变。
            "embedder": embedder_identity(self.embed_model, self.embed_dims),
            "extract_params": {"temperature": EXTRACT_TEMPERATURE, "max_tokens": EXTRACT_MAX_TOKENS, "top_p": self.memory.llm.config.top_p},
            "stored_count": self._stored_count(),
            "db_bytes": sum(files.values()),
            "db_files": files,
            "batches_ok": self.batches_ok,
            "batches_failed": self.batches_failed,
            "memories_added": self.memories_added,
        }

    def _stored_count(self) -> int:
        if self.closed:
            return self._final_count
        rows = self.memory.vector_store.list(filters={"user_id": BANK}, top_k=16_384)
        return len(rows[0]) if rows else 0

    def close(self) -> None:
        """Close milvus-lite and the history DB.
        关闭 milvus-lite 与历史库。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Releases file handles; stored_count is frozen.
            释放文件句柄；stored_count 冻结。
        """
        if self.closed:
            return
        self._final_count = self._stored_count()
        self.memory.vector_store.client.close()
        self.memory.db.close()
        self.closed = True


class EliteRealArm:
    """HMS Elite: SQLite store, deepseek-flash single-call extraction, fastembed bge-small; rows and card contexts; hybrid context (hms.context()).
    HMS Elite：SQLite 存储、deepseek-flash 单次抽取、fastembed bge-small；行上下文与卡片上下文；混合上下文（hms.context()）。
    """

    name = "elite"

    def __init__(
        self,
        workdir: Path,
        client: DeepSeekClient,
        meter: Meter,
        *,
        extraction_output: str | None = None,
        embed_model: str = EMBED_MODEL,
        embed_dims: int = EMBEDDING_DIMS,
        cjk: str | None = None,
        fastembed_cache_dir: str | None = None,
        entity_names: Sequence[str] = (),
    ) -> None:
        """HMS(HMSConfig(...)) then splice the accounting client into hms.llm and wrap the embedder.
        HMS(HMSConfig(...)) 后把记账客户端接入 hms.llm，并包装嵌入器。

        Input / 输入：
            workdir: Holds hms.db.
            workdir：存放 hms.db。

            client: Shared DeepSeekClient.
            client：共享 DeepSeekClient。

            meter: Shared ledger.
            meter：共享台账。

            extraction_output: IngestConfig.extraction_output ('full' | 'compact' | 'lines'); forwarded verbatim. None (the
                default) leaves the key out of the ingest section so the library default applies ('lines' since
                2026-09-18, CONTRACT §6.6 / changelog row 108); the harness keeps no default of its own.
            extraction_output：IngestConfig.extraction_output（'full' | 'compact' | 'lines'）；原样透传。None（默认）不写入
                ingest 段，由库默认值生效（2026-09-18 起为 'lines'，CONTRACT §6.6 / 变更记录第 108 行）；基准脚本自身不保留默认值。

            embed_model: fastembed model name (default EMBED_MODEL, the English run); the EgoLife-ZH runs pass BAAI/bge-small-zh-v1.5.
            embed_model：fastembed 模型名（默认 EMBED_MODEL，即英文运行）；EgoLife-ZH 运行传 BAAI/bge-small-zh-v1.5。

            embed_dims: The model's vector size (default EMBEDDING_DIMS); the library never validates it against the model, so the caller resolves it (run_real: fastembed.TextEmbedding.get_embedding_size).
            embed_dims：模型向量维数（默认 EMBEDDING_DIMS）；库不会对照模型校验，由调用方解析（run_real：fastembed.TextEmbedding.get_embedding_size）。

            cjk: None (default) leaves the library's `text` / `ingest.zh_rules` keys out so the defaults apply (English runs byte-identical); 'on' → text.cjk_tokens=bigram + ingest.zh_rules=True (run B); 'off' → text.cjk_tokens=none + ingest.zh_rules=False (run A, the English-centric baseline). CONTRACT §9.6.
            cjk：None（默认）不写库的 `text` / `ingest.zh_rules` 键，由默认值生效（英文运行逐字节不变）；'on' → text.cjk_tokens=bigram + ingest.zh_rules=True（运行 B）；'off' → text.cjk_tokens=none + ingest.zh_rules=False（运行 A，英文中心基线）。CONTRACT §9.6。

            fastembed_cache_dir: Persistent model cache handed to FastEmbedEmbedderConfig.cache_dir (None → fastembed's default / FASTEMBED_CACHE_PATH).
            fastembed_cache_dir：交给 FastEmbedEmbedderConfig.cache_dir 的持久模型缓存（None → fastembed 默认 / FASTEMBED_CACHE_PATH）。

            entity_names: Closed proper-name list for the rule entity extractor (RuleEntityExtractorConfig.names; the EgoLife-ZH glossary names); empty keeps the extractor config untouched.
            entity_names：规则实体抽取器的封闭人名表（RuleEntityExtractorConfig.names；EgoLife-ZH 术语表人名）；为空则不动抽取器配置。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Creates the SQLite file; loads the fastembed model.
            创建 SQLite 文件；加载 fastembed 模型。
        """
        if cjk not in (None, "on", "off"):
            raise ValueError(f"cjk must be None, 'on' or 'off', got {cjk!r}")
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.meter = meter
        self.embed_model = embed_model
        self.embed_dims = int(embed_dims)
        self.cjk = cjk
        self.db_path = workdir / "hms.db"
        embedder_config: dict[str, Any] = {"model": self.embed_model, "embedding_dims": self.embed_dims}
        if fastembed_cache_dir:
            embedder_config["cache_dir"] = str(fastembed_cache_dir)
        ingest: dict[str, Any] = {"extraction_mode": "single", **({} if extraction_output is None else {"extraction_output": extraction_output})}
        extra: dict[str, Any] = {}
        if cjk is not None:
            extra["text"] = {"cjk_tokens": "bigram" if cjk == "on" else "none"}
            ingest["zh_rules"] = cjk == "on"
        if entity_names:
            ingest["entity_extractor_config"] = {"names": [str(n) for n in entity_names]}
        self.config = HMSConfig(
            store={"path": str(self.db_path)},
            llm={"provider": "deepseek", "config": {"model": MODEL, "temperature": EXTRACT_TEMPERATURE, "max_tokens": EXTRACT_MAX_TOKENS}},
            embedder={"provider": "fastembed", "config": embedder_config},
            ingest=ingest,
            **extra,
        )
        self.hms = HMS(self.config)
        if not isinstance(self.hms.llm, EliteDeepSeekLLM):
            raise RuntimeError("HMS did not build its DeepSeek provider")
        self.hms.llm.client = client
        if self.config.ingest.extraction_output == "lines":
            self._reject_truncated_lines()
        instrument_embedder(self.hms.embedder, meter, self.name, client)
        self.batches_ok = 0
        self.batches_failed = 0
        self.rows_added = 0
        self.closed = False

    def _reject_truncated_lines(self) -> None:
        """Apply the harness truncation rule to the lines contract: a finish_reason == 'length' reply is handed to the
        library as '' so it stores nothing before the slice is bisected.
        把基准脚本的截断规则用于行式契约：finish_reason == 'length' 的回复以 '' 交给库，使切片被二分之前不存任何内容。

        Under the JSON contracts a cut reply is normally unparsable and stores nothing; under lines (CONTRACT §6.6)
        parse_lines keeps every complete line, so without this the parent slice's partial atoms would stay in the
        store next to the atoms of its two halves (duplicates, double-counted `added`). The raw reply is still in
        extract_raw_elite.jsonl; the empty string makes parse_lines log NO_SECTION_MARK, which the outcome ladder of
        retain_with_bisect ranks below `truncated`.
        JSON 契约下被截断的回复通常不可解析、什么都不存；行式契约（CONTRACT §6.6）下 parse_lines 会保留每个完整的行，
        若不如此处理，父切片的部分原子会与它两半的原子一起留在库里（重复、`added` 重复计数）。原始回复仍在
        extract_raw_elite.jsonl；空串使 parse_lines 记 NO_SECTION_MARK，而 retain_with_bisect 的结果阶梯把它排在
        `truncated` 之后。

        Input / 输入：
            none (wraps self.hms.llm.generate_lines; reads self.client.last_calls).
            无（包装 self.hms.llm.generate_lines；读取 self.client.last_calls）。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Replaces the bound generate_lines of this arm's LLM instance.
            替换该臂 LLM 实例上绑定的 generate_lines。
        """
        llm, client = self.hms.llm, self.client
        original = llm.generate_lines

        def generate_lines(task: str, *, system: str, user: str) -> str:
            text = original(task, system=system, user=user)
            last = client.last_calls[-1] if client.last_calls else None
            if last is not None and last.finish_reason == TRUNCATED:
                return ""
            return text

        llm.generate_lines = generate_lines  # type: ignore[method-assign]

    def _add(self, windows: list[Window], metadata: dict[str, Any], timestamp: str) -> dict[str, Any]:
        """One hms.add(<windows>, user_id=BANK, metadata=metadata, timestamp=<first window start>, infer=True).
        一次 hms.add(<windows>, user_id=BANK, metadata=metadata, timestamp=<首窗口起点>, infer=True)。

        Input / 输入：
            windows: Window slice.
            windows：窗口切片。

            metadata: Scalar metadata.
            metadata：标量元数据。

            timestamp: ISO instant of the first window.
            timestamp：首窗口的 ISO 时刻。

        Output / 输出：
            {'results': [row + 'event' ADD|UPDATE|NONE], 'downgraded': {verdict: count}} — the second key counts the
            NONE / REVISE / CONFLICT verdicts the ingest.verdict_min_similarity guard stored as ADD (HMS.last_downgraded).
            {'results': [行 + 'event' ADD|UPDATE|NONE], 'downgraded': {裁决: 数量}}——第二个键统计被
            ingest.verdict_min_similarity 守卫按 ADD 存入的 NONE / REVISE / CONFLICT 裁决（HMS.last_downgraded）。

        State effect / 状态影响：
            Writes the store.
            写库。
        """
        result = self.hms.add([{"role": "user", "content": w.text} for w in windows], user_id=BANK, metadata=metadata, timestamp=timestamp, infer=True)
        downgraded: dict[str, int] = {}
        for entry in self.hms.last_downgraded:
            key = str(entry.get("verdict", "?"))
            downgraded[key] = downgraded.get(key, 0) + 1
        return {**result, "downgraded": downgraded}

    def retain_batch(self, batch: Batch) -> dict[str, Any]:
        """retain_with_bisect(self, batch): hms.add(<=25 windows, ..., infer=True), halves re-submitted on truncation.
        retain_with_bisect(self, batch)：hms.add(<=25 个窗口, ..., infer=True)，截断时两半重提。

        Input / 输入：
            batch: The batch.
            batch：批次。

        Output / 输出：
            retain_with_bisect's record (added counts every add() result incl. UPDATE / NONE events; see 'events').
            retain_with_bisect 的记录（added 计入每个 add() 结果，含 UPDATE / NONE 事件；见 'events'）。

        State effect / 状态影响：
            Writes the store; counters batches_ok / batches_failed / rows_added updated.
            写库；更新 batches_ok / batches_failed / rows_added 计数。
        """
        row = retain_with_bisect(self, batch)
        if row["ok"]:
            self.batches_ok += 1
        else:
            self.batches_failed += 1
        self.rows_added += row["added"]
        return row

    def recall(self, question: str, query_iso: str) -> list[dict[str, Any]]:
        """search(question, top_k=20, threshold=0.1, reference_date=query_iso) -> rows in Elite order.
        search(question, top_k=20, threshold=0.1, reference_date=query_iso) -> 按 Elite 顺序的行。

        Input / 输入：
            question: Public question text.
            question：公开题干。

            query_iso: reference_date.
            query_iso：reference_date。

        Output / 输出：
            [{'batch_id', 'text', 'score', 'memory_id', 'keyword', 'entity', 'recalled_by', 'channel_sizes'}] — the last
            four come from search(explain=True)'s score_details (keyword = bm25_score, entity = entity_boost — the
            fused entity signal, the only entity key score_details carries; both 0.0 when the channel was inactive) and
            change neither the order nor the scores.
            [{'batch_id', 'text', 'score', 'memory_id', 'keyword', 'entity', 'recalled_by', 'channel_sizes'}]——后四个取自
            search(explain=True) 的 score_details（keyword = bm25_score，entity = entity_boost——融合后的实体信号，也是
            score_details 唯一的实体键；通道未激活时都为 0.0），不改变顺序与分数。

        State effect / 状态影响：
            One local query embedding.
            一次本地查询嵌入。
        """
        with self.client.label(arm=self.name, stage="recall", question_id=None, batch_id=None):
            result = self.hms.search(
                question, filters={"user_id": BANK}, top_k=TOP_K, threshold=THRESHOLD, reference_date=query_iso, explain=True
            )
        rows = []
        for row in result.get("results", []):
            meta = row.get("metadata") or {}
            details = row.get("score_details") or {}
            rows.append(
                {
                    "batch_id": meta.get("batch_id"),
                    "text": row.get("memory", ""),
                    "score": float(row.get("score", 0.0)),
                    "memory_id": row.get("id"),
                    "keyword": float(details.get("bm25_score", 0.0) or 0.0),
                    "entity": float(details.get("entity_boost", 0.0) or 0.0),
                    "recalled_by": list(details.get("recalled_by") or []),
                    "channel_sizes": dict(details.get("channel_sizes") or {}),
                }
            )
        return rows

    def card(self, question: str, query_iso: str) -> dict[str, Any]:
        """compile(question, filters, as_of=query_iso).to_text() plus item batch ids (via hms.get) and measurements.
        compile(question, filters, as_of=query_iso).to_text()，附各项目的批次 id（经 hms.get）与度量。

        Input / 输入：
            question: Public question text.
            question：公开题干。

            query_iso: as_of.
            query_iso：as_of。

        Output / 输出：
            {'text', 'chars', 'used_tokens', 'items', 'mode', 'compile_ms', 'batch_ids'}.
            {'text', 'chars', 'used_tokens', 'items', 'mode', 'compile_ms', 'batch_ids'}。

        State effect / 状态影响：
            Local only (compile has no LLM call; query embedding counted as 'recall').
            仅本地（compile 无 LLM 调用；查询嵌入计入 'recall'）。
        """
        t0 = time.perf_counter()
        with self.client.label(arm=self.name, stage="recall", question_id=None, batch_id=None):
            ir = self.hms.compile(question, filters={"user_id": BANK}, as_of=query_iso)
            text = ir.to_text()
        ms = (time.perf_counter() - t0) * 1000.0
        items = ir.items()
        batch_ids: list[str] = []
        for item in items:
            mid = getattr(item, "memory_id", None)
            if not mid:
                continue
            row = self.hms.get(mid)
            bid = ((row or {}).get("metadata") or {}).get("batch_id")
            if bid and bid not in batch_ids:
                batch_ids.append(bid)
        return {
            "text": text,
            "chars": len(text),
            "used_tokens": int(getattr(ir, "used_tokens", 0)),
            "items": len(items),
            "mode": str(getattr(getattr(ir, "mode", ""), "value", getattr(ir, "mode", ""))),
            "compile_ms": ms,
            "batch_ids": batch_ids,
        }

    def hybrid(self, question: str, query_iso: str) -> dict[str, Any]:
        """hms.context(question, filters, top_k=TOP_K, as_of=query_iso): the memoized session card + timestamped rows (CONTRACT §6.7), plus batch ids of both parts and measurements.
        hms.context(question, filters, top_k=TOP_K, as_of=query_iso)：记忆化会话卡片 + 带时间戳记忆行（CONTRACT §6.7），附两部分的批次 id 与度量。

        Input / 输入：
            question: Public question text.
            question：公开题干。

            query_iso: as_of of the rows (search reference_date).
            query_iso：记忆行的 as_of（search 的 reference_date）。

        Output / 输出：
            {'text', 'chars', 'context_ms', 'session_reused', 'session_key', 'session_chars', 'rows_kept', 'session_batch_ids', 'row_batch_ids', 'batch_ids', 'returned_count'}.
            {'text', 'chars', 'context_ms', 'session_reused', 'session_key', 'session_chars', 'rows_kept', 'session_batch_ids', 'row_batch_ids', 'batch_ids', 'returned_count'}。

        State effect / 状态影响：
            Local only (one search embedding counted as 'recall'; the card compiles only after an ingest).
            仅本地（一次查询嵌入计入 'recall'；卡片只在摄入之后重新编译）。
        """
        t0 = time.perf_counter()
        with self.client.label(arm=self.name, stage="recall", question_id=None, batch_id=None):
            ctx = self.hms.context(question, filters={"user_id": BANK}, top_k=TOP_K, as_of=query_iso)
        ms = (time.perf_counter() - t0) * 1000.0
        session_batch_ids: list[str] = []
        for mid in ctx["session_memory_ids"]:
            row = self.hms.get(mid)
            bid = ((row or {}).get("metadata") or {}).get("batch_id")
            if bid and bid not in session_batch_ids:
                session_batch_ids.append(bid)
        row_batch_ids: list[str] = []
        for row in ctx["rows"]:
            bid = (row.get("metadata") or {}).get("batch_id")
            if bid and bid not in row_batch_ids:
                row_batch_ids.append(bid)
        return {
            "text": ctx["text"],
            "chars": len(ctx["text"]),
            "context_ms": ms,
            "session_reused": bool(ctx["session_reused"]),
            "session_key": ctx["session_key"],
            "session_chars": len(ctx["session_card"]),
            "rows_kept": len(ctx["rows"]),
            "session_batch_ids": session_batch_ids,
            "row_batch_ids": row_batch_ids,
            "batch_ids": session_batch_ids + [b for b in row_batch_ids if b not in session_batch_ids],
            "returned_count": len(ctx["session_memory_ids"]) + len(ctx["rows"]),
        }

    def stats(self) -> dict[str, Any]:
        """Counts, sizes and the HMS usage meter (cross-check for the ledger).
        计数、尺寸与 HMS 用量计量器（与台账交叉核对）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with stored_count, db_bytes, hms_usage (exact when estimated=False), batches, rows_added.
            含 stored_count、db_bytes、hms_usage（estimated=False 时为精确值）、批次数、rows_added 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        usage = self.hms.usage()
        files = {
            self.db_path.name + suffix: _path_bytes(Path(str(self.db_path) + suffix))
            for suffix in ("", "-wal", "-shm")
            if Path(str(self.db_path) + suffix).exists()
        }
        return {
            "engine": "hms_elite HMS",
            "index": self.config.vector_store.provider,
            "scoring": self.config.scoring.model_dump(),
            "llm": self.hms.llm.identity,
            "embedder": self.hms.embedder.identity,
            "extract_params": {"temperature": EXTRACT_TEMPERATURE, "max_tokens": EXTRACT_MAX_TOKENS, "top_p": self.hms.llm.config.top_p},
            "extraction_mode": self.config.ingest.extraction_mode,
            "extraction_output": self.config.ingest.extraction_output,
            # CJK switches (CONTRACT §9.6): None = library defaults (the English runs); recorded so runs A / B are told apart.
            # CJK 开关（CONTRACT §9.6）：None = 库默认值（英文运行）；记录下来以区分运行 A / B。
            "cjk": self.cjk,
            "text_cjk_tokens": self.config.text.cjk_tokens,
            "zh_rules": self.config.ingest.zh_rules,
            "entity_extractor": self.hms.entity_extractor.identity,
            "compiler_budget_tokens": self.config.compiler.budget_tokens,
            "context_session_budget_tokens": self.config.compiler.context_session_budget_tokens,
            "context_rows_top_k": self.config.compiler.context_rows_top_k,
            "hms_usage": usage.to_dict(),
            "stored_count": self._stored_count(),
            "db_bytes": sum(files.values()),
            "db_files": files,
            "batches_ok": self.batches_ok,
            "batches_failed": self.batches_failed,
            "rows_added": self.rows_added,
            "count_note": "rows_added = add() results incl. UPDATE / NONE events (see retain events); stored_count = distinct rows in the store at close",
        }

    def _stored_count(self) -> int:
        if self.closed:
            return self._final_count
        return len(self.hms.get_all(filters={"user_id": BANK}, top_k=1_000_000).get("results", []))

    def close(self) -> None:
        """Close the HMS facade.
        关闭 HMS 门面。

        Input / 输入：
            none.
            无。

        Output / 输出：
            None.
            无。

        State effect / 状态影响：
            Checkpoints and closes SQLite; stored_count is frozen.
            检查点并关闭 SQLite；stored_count 冻结。
        """
        if self.closed:
            return
        self._final_count = self._stored_count()
        self.hms.close()
        self.closed = True


__all__ = [
    "ANSWER_MAX_TOKENS",
    "ANSWER_TEMPERATURE",
    "BANK",
    "BASE_URL",
    "BATCH_SIZE",
    "EMBEDDING_DIMS",
    "EMBED_MODEL",
    "ENGINE_NAMES",
    "embedder_identity",
    "EXTRACT_MAX_TOKENS",
    "EXTRACT_TEMPERATURE",
    "LABELS",
    "LogCapture",
    "MAX_RETRIES",
    "BACKOFF_S",
    "RETRY_STATUSES",
    "THINKING",
    "THRESHOLD",
    "TIMEOUT_S",
    "TOP_K",
    "AnswerResult",
    "Batch",
    "DeepSeekClient",
    "EliteRealArm",
    "Mem0RealArm",
    "TRUNCATED",
    "answer",
    "build_question_prompt",
    "instrument_embedder",
    "load_api_key",
    "parse_choice",
    "plan_batches",
    "retain_with_bisect",
    "rows_to_context",
    "sub_metadata",
    "their_protocol_digest",
]
