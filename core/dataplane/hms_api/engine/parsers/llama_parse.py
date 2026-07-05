"""LlamaParse parser implementation using the LlamaIndex Cloud parsing API."""

import asyncio
import logging
import mimetypes
import time

import httpx

from .base import FileParser, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

_LLAMA_PARSE_BASE_URL = "https://api.cloud.llamaindex.ai/api/parsing"
_DEFAULT_POLL_INTERVAL = 2.0  # seconds
_DEFAULT_TIMEOUT = 300.0  # seconds

# HTTP status codes that indicate the file type is not supported.
# Other 4xx codes (401, 403, 429, etc.) are operational errors, not file-type issues.
_UNSUPPORTED_FILE_STATUS_CODES = {400, 415, 422}


class LlamaParseParser(FileParser):
    """
    LlamaParse file parser using LlamaIndex's hosted parsing service.

    Uploads files to the LlamaParse API, polls until the parse job completes,
    and returns the resulting markdown. The API determines which file types
    are supported — UnsupportedFileTypeError is raised if the file is rejected.
    """

    def __init__(
        self,
        api_key: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        """
        Initialize llama_parse parser.

        Args:
            api_key: LlamaCloud API key (typically starts with "llx-")
            poll_interval: Seconds between status poll requests (default: 2)
            timeout: Maximum seconds to wait for parsing (default: 300)
        """
        self._api_key = api_key
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._auth_headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0))

    async def convert(self, file_data: bytes, filename: str) -> str:
        """
        Parse file to markdown using the LlamaParse API.

        Raises:
            UnsupportedFileTypeError: If the LlamaParse API rejects the file type
            RuntimeError: If parsing fails for another reason
        """
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        # Step 1: Upload file and start parse job
        upload_resp = await self._client.post(
            f"{_LLAMA_PARSE_BASE_URL}/upload",
            headers=self._auth_headers,
            # Ensure file_data is plain bytes (storage backends may return obstore.Bytes)
            files={"file": (filename, bytes(file_data), content_type)},
        )
        _raise_for_status(upload_resp, filename, "upload")
        job_id: str = upload_resp.json()["id"]

        # Step 2: Poll job status until SUCCESS or ERROR
        deadline = time.monotonic() + self._timeout
        while True:
            status_resp = await self._client.get(
                f"{_LLAMA_PARSE_BASE_URL}/job/{job_id}",
                headers=self._auth_headers,
            )
            _raise_for_status(status_resp, filename, "poll job status")
            status_data = status_resp.json()
            status = status_data.get("status")

            if status == "SUCCESS":
                break
            if status in ("ERROR", "CANCELLED"):
                error = status_data.get("error_code") or status_data.get("error") or "unknown error"
                raise RuntimeError(f"LlamaParse job failed for '{filename}': {error}")

            if time.monotonic() >= deadline:
                raise RuntimeError(f"LlamaParse job timed out after {self._timeout}s for '{filename}'")

            await asyncio.sleep(self._poll_interval)

        # Step 3: Fetch the markdown result
        result_resp = await self._client.get(
            f"{_LLAMA_PARSE_BASE_URL}/job/{job_id}/result/markdown",
            headers=self._auth_headers,
        )
        _raise_for_status(result_resp, filename, "fetch markdown result")
        markdown = result_resp.json().get("markdown")
        if not markdown:
            raise RuntimeError(f"No content extracted from '{filename}'")
        return markdown

    def name(self) -> str:
        """Get parser name."""
        return "llama_parse"


def _raise_for_status(response: httpx.Response, filename: str, step: str) -> None:
    """
    Raise an appropriate error for HTTP errors.

    Raises UnsupportedFileTypeError for 400/415/422 (file rejected by the API).
    Raises RuntimeError for all other errors (auth, rate-limit, server errors).
    """
    if not response.is_error:
        return
    body = response.text or "<empty>"
    msg = (
        f"LlamaParse API error during {step} for '{filename}': {response.status_code} {response.reason_phrase} — {body}"
    )
    if response.status_code in _UNSUPPORTED_FILE_STATUS_CODES:
        raise UnsupportedFileTypeError(msg)
    raise RuntimeError(msg)
