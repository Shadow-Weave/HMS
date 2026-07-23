"""Abstract and backwards-compatible contracts for file parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hms_api.engine.multimodal.checkpoints import VideoSegmentCheckpoint, VideoSegmentIdentity


class UnsupportedFileTypeError(Exception):
    """Raised by a parser when it does not support the given file type."""

    pass


class ParserNotApplicableError(UnsupportedFileTypeError):
    """Typed pre-processing outcome that permits an explicit fallback parser."""


class ParserProcessingError(RuntimeError):
    """Sanitized terminal failure after a strict parser started processing."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        logical_calls: int = 0,
        physical_attempts: int = 0,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.logical_calls = logical_calls
        self.physical_attempts = physical_attempts


@dataclass(frozen=True)
class ConversionInput:
    """Typed input passed to parsers that need MIME and asset provenance."""

    file_data: bytes = field(repr=False)
    filename: str
    content_type: str | None = None
    asset_id: str | None = None
    asset_sha256: str | None = None
    source_available: bool = True
    before_provider_call: Callable[[], Awaitable[None]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    load_video_segment_checkpoint: Callable[[VideoSegmentIdentity], Awaitable[VideoSegmentCheckpoint | None]] | None = (
        field(default=None, repr=False, compare=False)
    )
    save_video_segment_checkpoint: Callable[[VideoSegmentCheckpoint], Awaitable[None]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ParserOutput:
    """Richer parser output while preserving the legacy string seam."""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, str | None]] = field(default_factory=list)
    retain_extraction_mode: str | None = None
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)


class FileParser(ABC):
    """Abstract base for file to markdown parsers."""

    @abstractmethod
    async def convert(self, file_data: bytes, filename: str) -> str:
        """
        Parse file to markdown.

        Args:
            file_data: Raw file bytes
            filename: Original filename (used for format detection)

        Returns:
            Markdown content as string

        Raises:
            UnsupportedFileTypeError: If the file type is not supported by this parser
            RuntimeError: If parsing fails for another reason
        """
        pass

    async def convert_input(self, request: ConversionInput) -> ParserOutput:
        """Typed adapter used by the registry.

        Existing parsers inherit this implementation and therefore keep their
        exact ``convert(bytes, filename) -> str`` behavior.
        """

        return ParserOutput(content=await self.convert(request.file_data, request.filename))

    @property
    def terminal_processing_failures(self) -> bool:
        """Whether unexpected failures after selection are terminal.

        Legacy parsers retain the historical catch-all fallback behavior.
        Privacy/cost-sensitive multimodal parsers override this to ``True``.
        """

        return False

    def supports(self, filename: str, content_type: str | None = None) -> bool:
        """
        Check if parser supports this file type.

        Override this for local/static extension-based filtering.
        Parsers that delegate to a remote service should leave this as True
        and raise UnsupportedFileTypeError from convert() instead.

        Args:
            filename: File name (used for extension check)
            content_type: MIME type (optional)

        Returns:
            True if this parser can handle the file (default: True)
        """
        return True

    @abstractmethod
    def name(self) -> str:
        """
        Get parser name.

        Returns:
            Parser name (e.g., "markitdown")
        """
        pass
