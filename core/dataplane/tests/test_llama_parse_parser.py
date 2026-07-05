"""
Tests for the LlamaParse file parser.

Unit tests run always (mocked HTTP). Integration tests require
HMS_API_FILE_PARSER_LLAMA_PARSE_API_KEY in the environment.
"""

import json
import os
from unittest.mock import AsyncMock

import httpx
import pytest

from hms_api.config import ENV_FILE_PARSER_LLAMA_PARSE_API_KEY
from hms_api.engine.parsers.base import UnsupportedFileTypeError
from hms_api.engine.parsers.llama_parse import LlamaParseParser

_api_key = os.getenv(ENV_FILE_PARSER_LLAMA_PARSE_API_KEY)

# Minimal valid PDF with the text "Hello from HMS"
_SAMPLE_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello from HMS) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
369
%%EOF"""


def _mock_response(status_code: int, json_data: dict | None = None, text: str = "") -> httpx.Response:
    """Build a fake httpx.Response."""
    content = json.dumps(json_data).encode() if json_data is not None else text.encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        request=httpx.Request("GET", "https://fake"),
    )


# ---------------------------------------------------------------------------
# Unit tests (always run — mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_success():
    """Happy path: upload → poll SUCCESS → fetch markdown."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=10.0)

    upload_resp = _mock_response(200, {"id": "job-123"})
    poll_resp = _mock_response(200, {"status": "SUCCESS"})
    result_resp = _mock_response(200, {"markdown": "# Hello"})

    parser._client = AsyncMock()
    parser._client.post = AsyncMock(return_value=upload_resp)
    parser._client.get = AsyncMock(side_effect=[poll_resp, result_resp])

    result = await parser.convert(b"fake-pdf", "test.pdf")
    assert result == "# Hello"


@pytest.mark.asyncio
async def test_convert_polls_until_success():
    """Parser should poll multiple times before SUCCESS."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=10.0)

    upload_resp = _mock_response(200, {"id": "job-456"})
    pending_resp = _mock_response(200, {"status": "PENDING"})
    success_resp = _mock_response(200, {"status": "SUCCESS"})
    result_resp = _mock_response(200, {"markdown": "parsed content"})

    parser._client = AsyncMock()
    parser._client.post = AsyncMock(return_value=upload_resp)
    parser._client.get = AsyncMock(side_effect=[pending_resp, pending_resp, success_resp, result_resp])

    result = await parser.convert(b"fake", "doc.pdf")
    assert result == "parsed content"
    assert parser._client.get.call_count == 4  # 2 pending + 1 success + 1 result


@pytest.mark.asyncio
async def test_convert_job_error():
    """Parser should raise RuntimeError when job status is ERROR."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=10.0)

    upload_resp = _mock_response(200, {"id": "job-err"})
    error_resp = _mock_response(200, {"status": "ERROR", "error_code": "PARSE_FAILED"})

    parser._client = AsyncMock()
    parser._client.post = AsyncMock(return_value=upload_resp)
    parser._client.get = AsyncMock(return_value=error_resp)

    with pytest.raises(RuntimeError, match="PARSE_FAILED"):
        await parser.convert(b"bad", "bad.pdf")


@pytest.mark.asyncio
async def test_convert_timeout():
    """Parser should raise RuntimeError on timeout."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=0.0)

    upload_resp = _mock_response(200, {"id": "job-slow"})
    pending_resp = _mock_response(200, {"status": "PENDING"})

    parser._client = AsyncMock()
    parser._client.post = AsyncMock(return_value=upload_resp)
    parser._client.get = AsyncMock(return_value=pending_resp)

    with pytest.raises(RuntimeError, match="timed out"):
        await parser.convert(b"data", "slow.pdf")


@pytest.mark.asyncio
async def test_upload_unsupported_file_type():
    """400/415/422 on upload should raise UnsupportedFileTypeError."""
    for status_code in (400, 415, 422):
        parser = LlamaParseParser(api_key="llx-test")
        reject_resp = _mock_response(status_code, text="unsupported format")

        parser._client = AsyncMock()
        parser._client.post = AsyncMock(return_value=reject_resp)

        with pytest.raises(UnsupportedFileTypeError):
            await parser.convert(b"data", "file.xyz")


@pytest.mark.asyncio
async def test_auth_error_raises_runtime_error():
    """401/403 should raise RuntimeError, not UnsupportedFileTypeError."""
    for status_code in (401, 403):
        parser = LlamaParseParser(api_key="bad-key")
        auth_resp = _mock_response(status_code, text="unauthorized")

        parser._client = AsyncMock()
        parser._client.post = AsyncMock(return_value=auth_resp)

        with pytest.raises(RuntimeError, match="unauthorized"):
            await parser.convert(b"data", "file.pdf")


@pytest.mark.asyncio
async def test_rate_limit_raises_runtime_error():
    """429 should raise RuntimeError, not UnsupportedFileTypeError."""
    parser = LlamaParseParser(api_key="llx-test")
    rate_resp = _mock_response(429, text="rate limited")

    parser._client = AsyncMock()
    parser._client.post = AsyncMock(return_value=rate_resp)

    with pytest.raises(RuntimeError, match="rate limited"):
        await parser.convert(b"data", "file.pdf")


def test_parser_name():
    """LlamaParseParser.name() should return 'llama_parse'."""
    parser = LlamaParseParser(api_key="llx-test")
    assert parser.name() == "llama_parse"


# ---------------------------------------------------------------------------
# Integration tests (require API key)
# ---------------------------------------------------------------------------

_integration = pytest.mark.skipif(
    not _api_key,
    reason="HMS_API_FILE_PARSER_LLAMA_PARSE_API_KEY not set",
)


@pytest.fixture
def llama_parse_parser() -> LlamaParseParser:
    assert _api_key is not None
    return LlamaParseParser(api_key=_api_key)


@_integration
@pytest.mark.asyncio
async def test_llama_parse_parser_converts_pdf(llama_parse_parser: LlamaParseParser):
    """LlamaParseParser should extract text from a valid PDF."""
    result = await llama_parse_parser.convert(_SAMPLE_PDF, "sample.pdf")
    assert isinstance(result, str)
    assert len(result) > 0
