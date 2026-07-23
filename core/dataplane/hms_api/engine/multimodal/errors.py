"""Typed, sanitized errors for multimodal ingestion.

The exception messages in this module are intentionally content-free.  Raw
provider responses, prompts, file bytes, and data URLs must never be attached
to these exceptions because worker tracebacks are persisted and exported.
"""

from __future__ import annotations


class MultimodalError(RuntimeError):
    """Base error with a stable public code and retry classification."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        logical_calls: int = 0,
        physical_attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.logical_calls = logical_calls
        self.physical_attempts = physical_attempts


class MediaValidationError(MultimodalError):
    """The uploaded media failed deterministic local validation."""


class MediaBudgetExceededError(MediaValidationError):
    """The media exceeds a configured pre-provider resource budget."""


class AnimatedImageNotSupportedError(MediaValidationError):
    """Animated images are not silently truncated to their first frame."""


class VideoDecoderUnavailableError(MultimodalError):
    """The optional local video decoder is not installed or loadable."""


class VideoDecodeError(MediaValidationError):
    """The uploaded video could not be safely probed or decoded."""


class ProviderError(MultimodalError):
    """Base error for the isolated multimodal description provider."""


class ProviderAuthenticationError(ProviderError):
    """Provider credentials or model access are invalid."""


class ProviderRateLimitError(ProviderError):
    """Provider throttled the request after bounded retries."""


class ProviderUnavailableError(ProviderError):
    """Transient provider/network failure after bounded retries."""


class ProviderRefusalError(ProviderError):
    """The model returned a safety refusal; this is not a schema repair case."""


class ProviderIncompleteError(ProviderError):
    """The Responses API returned an incomplete response."""


class ProviderSchemaError(ProviderError):
    """Provider output failed the strict local schema/grounding contract."""


class GroundingError(MultimodalError):
    """A semantic statement references missing or invalid visual evidence."""
