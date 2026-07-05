"""Tests for lib/client.py — HMS REST API client."""

from unittest.mock import patch

from conftest import FakeHTTPResponse

from lib.client import USER_AGENT, HMSClient


class TestUserAgentHeader:
    """Regression tests for #1041.

    The stdlib default ``Python-urllib/X.Y`` UA is blocked by Cloudflare with
    error 1010, so every request must carry our identifying UA.
    """

    def test_recall_sends_user_agent(self):
        c = HMSClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.recall("bank", "query")

        assert captured["ua"] == USER_AGENT
        assert captured["ua"].startswith("hms-codex/")

    def test_health_check_sends_user_agent(self):
        c = HMSClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({}, status=200)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.health_check(timeout=1)

        assert captured["ua"] == USER_AGENT
