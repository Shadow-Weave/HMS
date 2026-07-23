"""File parser implementations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hms_api.engine.multimodal.checkpoints import VideoSegmentCheckpoint, VideoSegmentIdentity

from .base import (
    ConversionInput,
    FileParser,
    ParserNotApplicableError,
    ParserOutput,
    ParserProcessingError,
    UnsupportedFileTypeError,
)
from .iris import IrisParser
from .llama_parse import LlamaParseParser
from .markitdown import MarkitdownParser

__all__ = [
    "FileParser",
    "UnsupportedFileTypeError",
    "ParserNotApplicableError",
    "ParserProcessingError",
    "ConversionInput",
    "ParserOutput",
    "IrisParser",
    "LlamaParseParser",
    "MarkitdownParser",
    "MultimodalParserConfig",
    "OpenAIMultimodalParser",
    "create_openai_multimodal_parser",
    "FileParserRegistry",
    "ConvertResult",
]


_MULTIMODAL_EXPORTS = {
    "MultimodalParserConfig",
    "OpenAIMultimodalParser",
    "create_openai_multimodal_parser",
}


def __getattr__(name: str) -> Any:
    """Load the optional media stack only when a caller asks for it.

    ``parsers`` is imported on every file-enabled HMS startup.  Keeping the
    multimodal implementation behind this PEP 562 hook prevents ordinary text
    and legacy file paths from importing Pillow/PyAV or constructing a vision
    provider while preserving the existing package-level import surface.
    """

    if name not in _MULTIMODAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import openai_multimodal

    value = getattr(openai_multimodal, name)
    globals()[name] = value
    return value


@dataclass
class ConvertResult:
    """Result of a successful file conversion."""

    content: str
    parser_name: str
    metadata: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, str | None]] = field(default_factory=list)
    retain_extraction_mode: str | None = None
    pipeline_metadata: dict = field(default_factory=dict)


logger = logging.getLogger(__name__)


class FileParserRegistry:
    """Registry for file parsers with auto-detection."""

    def __init__(self):
        """Initialize empty parser registry."""
        self._parsers: dict[str, FileParser] = {}

    def register(self, parser: FileParser):
        """
        Register a parser.

        Args:
            parser: FileParser instance
        """
        self._parsers[parser.name()] = parser

    def get_parser(
        self,
        name: str | None,
        filename: str,
        content_type: str | None = None,
    ) -> FileParser:
        """
        Get parser by name or auto-detect.

        Args:
            name: Parser name (e.g., "markitdown") or None for auto-detect
            filename: File name for auto-detection
            content_type: MIME type (optional)

        Returns:
            FileParser instance

        Raises:
            ValueError: If no suitable parser found
        """
        if name:
            # Explicit parser requested — return it directly, let the parser
            # raise UnsupportedFileTypeError from convert() if needed
            if name not in self._parsers:
                raise ValueError(f"Parser '{name}' not found. Available: {list(self._parsers.keys())}")
            return self._parsers[name]

        # Auto-detect parser
        for parser in self._parsers.values():
            if parser.supports(filename, content_type):
                return parser

        raise ValueError(f"No parser found for {filename}. Available parsers: {list(self._parsers.keys())}")

    async def convert_with_fallback(
        self,
        parsers: list[str],
        file_data: bytes,
        filename: str,
        content_type: str | None = None,
        *,
        asset_id: str | None = None,
        asset_sha256: str | None = None,
        source_available: bool = True,
        before_provider_call: Callable[[], Awaitable[None]] | None = None,
        load_video_segment_checkpoint: Callable[[VideoSegmentIdentity], Awaitable[VideoSegmentCheckpoint | None]]
        | None = None,
        save_video_segment_checkpoint: Callable[[VideoSegmentCheckpoint], Awaitable[None]] | None = None,
    ) -> ConvertResult:
        """
        Try each parser in order, falling back on failure or empty content.

        Moves to the next parser if the current one raises UnsupportedFileTypeError
        or returns empty content. Any other exception (RuntimeError, network error,
        etc.) also triggers a fallback so the chain is exhausted before failing.

        Args:
            parsers: Ordered list of parser names to try
            file_data: Raw file bytes
            filename: Original filename
            content_type: MIME type (optional)

        Returns:
            ConvertResult with the parsed content and the name of the parser that succeeded

        Raises:
            ValueError: If a parser name is not registered
            RuntimeError: If all parsers fail or return empty content
        """
        last_error: Exception | None = None
        for name in parsers:
            parser = self.get_parser(name, filename, content_type)
            try:
                output = await parser.convert_input(
                    ConversionInput(
                        file_data=file_data,
                        filename=filename,
                        content_type=content_type,
                        asset_id=asset_id,
                        asset_sha256=asset_sha256,
                        source_available=source_available,
                        before_provider_call=before_provider_call,
                        load_video_segment_checkpoint=load_video_segment_checkpoint,
                        save_video_segment_checkpoint=save_video_segment_checkpoint,
                    )
                )
                if output.content and output.content.strip():
                    return ConvertResult(
                        content=output.content,
                        parser_name=name,
                        metadata=output.metadata,
                        entities=output.entities,
                        retain_extraction_mode=output.retain_extraction_mode,
                        pipeline_metadata=output.pipeline_metadata,
                    )
                logger.warning(f"Parser '{name}' returned empty content for '{filename}', trying next")
                if parser.terminal_processing_failures:
                    raise ParserProcessingError("parser.empty_output", f"Parser '{name}' returned no content")
                last_error = RuntimeError(f"Parser '{name}' returned no content for '{filename}'")
            except UnsupportedFileTypeError as e:
                logger.warning(f"Parser '{name}' does not support '{filename}', trying next: {e}")
                last_error = e
            except ParserProcessingError:
                raise
            except Exception as e:
                if parser.terminal_processing_failures:
                    raise ParserProcessingError(
                        "parser.processing_failed",
                        f"Parser '{name}' failed after processing started",
                        retryable=getattr(e, "retryable", False),
                        logical_calls=getattr(e, "logical_calls", 0),
                        physical_attempts=getattr(e, "physical_attempts", 0),
                    ) from None
                logger.warning(f"Parser '{name}' failed for '{filename}', trying next: {e}")
                last_error = e

        raise last_error or RuntimeError(f"No parsers available for '{filename}'")

    def list_parsers(self) -> list[str]:
        """Get list of registered parser names."""
        return list(self._parsers.keys())
