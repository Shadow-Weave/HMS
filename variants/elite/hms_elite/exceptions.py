"""Structured exception hierarchy with Mem0-compatible attribute names and one error-code table.
带 Mem0 兼容属性名与单一错误码表的结构化异常层次。
"""

# Purpose (EN): Define every error HMS Elite raises and the helper that unwraps them from pydantic errors.
# 目的（中）：定义 HMS Elite 抛出的全部错误，以及从 pydantic 错误中解包它们的辅助函数。

# Input (EN): Messages, error codes, detail dicts.
# 输入（中）：消息、错误码、细节字典。

# Output (EN): Exception instances carrying message/error_code/details/suggestion/debug_info.
# 输出（中）：携带 message/error_code/details/suggestion/debug_info 的异常实例。

# State effect (EN): None.
# 状态影响（中）：无。

from __future__ import annotations

from typing import Any


class HMSError(Exception):
    """Base error: a message plus a structured error code and optional details.
    基础错误：消息加结构化错误码与可选细节。
    """

    default_code = "HMS_000"
    default_suggestion: str | None = None

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict | None = None,
        suggestion: str | None = None,
        debug_info: dict | None = None,
    ) -> None:
        """Build the error and store its structured attributes.
        构造错误并保存其结构化属性。

        Input / 输入：
            message: Human-readable description.
            message：可读描述。

            error_code: Code from the CONTRACT §0.2 table; defaults per subclass.
            error_code：契约 §0.2 表中的错误码；各子类有默认值。

            details: Structured context (ids, keys, offending values).
            details：结构化上下文（id、键、非法值）。

            suggestion: Remediation hint for the caller.
            suggestion：给调用方的修复提示。

            debug_info: Internal diagnostic data.
            debug_info：内部诊断数据。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.default_code
        self.details = dict(details) if details else {}
        self.suggestion = suggestion if suggestion is not None else self.default_suggestion
        self.debug_info = dict(debug_info) if debug_info else {}

    def __repr__(self) -> str:  # noqa: D105 - dunder
        return f"{type(self).__name__}(message={self.message!r}, error_code={self.error_code!r})"

    def __str__(self) -> str:  # noqa: D105 - dunder
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error to a plain dict (Mem0 MemoryError.to_dict shape).
        将错误序列化为普通字典（Mem0 MemoryError.to_dict 形状）。

        Input / 输入：
            none.
            无。

        Output / 输出：
            Dict with error_type, error_code, message, details, suggestion, debug_info.
            含 error_type、error_code、message、details、suggestion、debug_info 的字典。

        State effect / 状态影响：
            None.
            无。
        """
        return {
            "error_type": type(self).__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": dict(self.details),
            "suggestion": self.suggestion,
            "debug_info": dict(self.debug_info),
        }


class ValidationError(HMSError, ValueError):
    """Input validation failure; also a ValueError so Mem0-ported callers keep working.
    输入校验失败；同时是 ValueError，使移植自 Mem0 的调用方继续可用。
    """

    default_code = "VALIDATION_000"


class ConfigError(ValidationError):
    """Configuration problem (CONFIG_001 pydantic, CONFIG_002 provider, CONFIG_003 removed key).
    配置问题（CONFIG_001 pydantic、CONFIG_002 提供者、CONFIG_003 已移除键）。
    """

    default_code = "CONFIG_001"


class LLMError(HMSError):
    """LLM transport or contract failure.
    LLM 传输或契约失败。
    """

    default_code = "LLM_001"
    default_suggestion = "Please check your LLM configuration and API key"


class EmbeddingError(HMSError):
    """Embedding provider failure.
    嵌入提供者失败。
    """

    default_code = "EMBED_001"


class StoreError(HMSError):
    """SQLite failure or invariant trigger (STORE_001/002/003).
    SQLite 失败或不变量触发器（STORE_001/002/003）。
    """

    default_code = "STORE_001"


class WitnessMissingError(StoreError):
    """A state-bearing object was written without a valid anchor (No Witness, No Meaning).
    承载状态的对象在写入时缺少有效锚点（无证据即无意义）。
    """

    default_code = "WITNESS_001"
    default_suggestion = "Every HMS object must reference at least one anchor_id; ingest the evidence first"


class ProofObligationError(HMSError):
    """A PC-IR proof obligation was violated.
    PC-IR 证明义务被违反。
    """

    default_code = "PCIR_001"


class DependencyError(HMSError):
    """An optional SDK is not installed; details carry the package and pip extra.
    可选 SDK 未安装；details 携带包名与 pip extra。
    """

    default_code = "DEPS_001"

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict | None = None,
        suggestion: str | None = None,
        debug_info: dict | None = None,
    ) -> None:
        """Build the error; the suggestion defaults to the pip install hint for details['extra'].
        构造错误；suggestion 默认为 details['extra'] 对应的 pip 安装提示。

        Input / 输入：
            message: Human-readable description.
            message：可读描述。

            error_code: Error code; defaults to DEPS_001.
            error_code：错误码；默认 DEPS_001。

            details: {'package': str, 'extra': str}.
            details：{'package': str, 'extra': str}。

            suggestion: Override of the default pip hint.
            suggestion：覆盖默认 pip 提示。

            debug_info: Internal diagnostic data.
            debug_info：内部诊断数据。

        Output / 输出：
            None (constructor).
            无（构造函数）。

        State effect / 状态影响：
            None.
            无。
        """
        extra = (details or {}).get("extra")
        if suggestion is None and extra:
            suggestion = f"pip install 'hms-elite[{extra}]'"
        super().__init__(message, error_code, details, suggestion, debug_info)


def unwrap_hms_error(exc: BaseException) -> HMSError | None:
    """Find the first HMSError hidden inside a pydantic ValidationError (validators raise ValueError subclasses).
    从 pydantic ValidationError 中找出第一个被包裹的 HMSError（校验器抛出的 ValueError 子类）。

    Input / 输入：
        exc: Any exception; pydantic errors are inspected via errors()[i]['ctx']['error'].
        exc：任意异常；对 pydantic 错误检查 errors()[i]['ctx']['error']。

    Output / 输出：
        The embedded HMSError, the exception itself when it already is one, else None.
        被包裹的 HMSError；若本身就是 HMSError 则返回其自身；否则 None。

    State effect / 状态影响：
        None.
        无。
    """
    if isinstance(exc, HMSError):
        return exc
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return None
    try:
        entries = errors()
    except Exception:  # pragma: no cover - defensive
        return None
    for entry in entries or ():
        ctx = entry.get("ctx") if isinstance(entry, dict) else None
        inner = ctx.get("error") if isinstance(ctx, dict) else None
        if isinstance(inner, HMSError):
            return inner
    return None


__all__ = [
    "ConfigError",
    "DependencyError",
    "EmbeddingError",
    "HMSError",
    "LLMError",
    "ProofObligationError",
    "StoreError",
    "ValidationError",
    "WitnessMissingError",
    "unwrap_hms_error",
]
