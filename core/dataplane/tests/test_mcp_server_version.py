"""Tests for MCP server identity reported via serverInfo."""

from unittest.mock import MagicMock

from hms_api import __version__ as HMS_VERSION
from hms_api.api.mcp import create_mcp_server


def test_mcp_server_reports_hms_version():
    """serverInfo.version should be HMS's version, not the FastMCP library version."""
    memory = MagicMock()
    server = create_mcp_server(memory, multi_bank=True)
    assert server.version == HMS_VERSION
