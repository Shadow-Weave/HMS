"""Compatibility and terminal-failure contracts for parser fallback."""

from collections.abc import Awaitable, Callable

import pytest

from hms_api.engine.parsers import (
    ConversionInput,
    FileParser,
    FileParserRegistry,
    ParserNotApplicableError,
    ParserOutput,
    ParserProcessingError,
)


class _Parser(FileParser):
    def __init__(
        self,
        parser_name: str,
        convert: Callable[[ConversionInput], Awaitable[ParserOutput]],
        *,
        strict: bool = False,
    ) -> None:
        self._parser_name = parser_name
        self._convert = convert
        self._strict = strict
        self.calls = 0

    async def convert(self, file_data: bytes, filename: str) -> str:
        raise AssertionError("registry must use the typed convert_input adapter")

    async def convert_input(self, request: ConversionInput) -> ParserOutput:
        self.calls += 1
        return await self._convert(request)

    @property
    def terminal_processing_failures(self) -> bool:
        return self._strict

    def supports(self, filename: str, content_type: str | None = None) -> bool:
        return True

    def name(self) -> str:
        return self._parser_name


def _registry(*parsers: FileParser) -> FileParserRegistry:
    registry = FileParserRegistry()
    for parser in parsers:
        registry.register(parser)
    return registry


async def _success(_: ConversionInput) -> ParserOutput:
    return ParserOutput(content="complete fallback output")


@pytest.mark.asyncio
async def test_legacy_runtime_error_preserves_historical_fallback() -> None:
    async def fail(_: ConversionInput) -> ParserOutput:
        raise RuntimeError("legacy parser failed")

    legacy = _Parser("legacy", fail)
    fallback = _Parser("fallback", _success)

    result = await _registry(legacy, fallback).convert_with_fallback(["legacy", "fallback"], b"input", "sample.bin")

    assert result.content == "complete fallback output"
    assert result.parser_name == "fallback"
    assert legacy.calls == fallback.calls == 1


@pytest.mark.asyncio
async def test_strict_not_applicable_can_fall_through_before_processing() -> None:
    async def not_applicable(_: ConversionInput) -> ParserOutput:
        raise ParserNotApplicableError("detected media type is not visual")

    multimodal = _Parser("openai_multimodal", not_applicable, strict=True)
    fallback = _Parser("fallback", _success)

    result = await _registry(multimodal, fallback).convert_with_fallback(
        ["openai_multimodal", "fallback"], b"input", "sample.bin"
    )

    assert result.parser_name == "fallback"
    assert multimodal.calls == fallback.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_retryable"),
    [
        (ParserProcessingError("provider.refusal", "request refused"), "provider.refusal", False),
        (TimeoutError("provider timed out"), "parser.processing_failed", False),
    ],
)
async def test_strict_processing_failure_is_terminal_and_never_mixes_partial_output(
    failure: Exception,
    expected_code: str,
    expected_retryable: bool,
) -> None:
    async def fail_after_partial_work(_: ConversionInput) -> ParserOutput:
        # Local state represents a segment result that must never escape when a
        # later required segment/provider step fails.
        partial_output = ParserOutput(content="uncommitted partial segment")
        assert partial_output.content
        raise failure

    multimodal = _Parser("openai_multimodal", fail_after_partial_work, strict=True)
    fallback = _Parser("fallback", _success)

    with pytest.raises(ParserProcessingError) as exc_info:
        await _registry(multimodal, fallback).convert_with_fallback(
            ["openai_multimodal", "fallback"], b"input", "sample.png", "image/png"
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is expected_retryable
    assert multimodal.calls == 1
    assert fallback.calls == 0
    assert "uncommitted partial segment" not in str(exc_info.value)
