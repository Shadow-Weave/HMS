"""Tests for daemonize() — subprocess.Popen re-exec instead of os.fork()."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_daemonize_parent_reexecs_via_popen(monkeypatch, tmp_path):
    """Parent path: daemonize() must spawn a child via subprocess.Popen and exit."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("_HMS_DAEMON_CHILD", raising=False)
    monkeypatch.setattr(sys, "argv", ["hms-api", "--daemon", "--port", "9999"])

    log_path = tmp_path / "daemon.log"
    monkeypatch.setattr("hms_api.daemon.DAEMON_LOG_PATH", log_path)

    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        proc = MagicMock()
        proc.pid = 99999
        return proc

    with (
        patch("hms_api.daemon.subprocess.Popen", side_effect=fake_popen),
        pytest.raises(SystemExit) as exc_info,
    ):
        from hms_api.daemon import daemonize

        daemonize()

    assert exc_info.value.code == 0

    # Verify child command does NOT contain --daemon
    assert "--daemon" not in captured["cmd"]
    # Verify it uses the module entry point
    assert "-m" in captured["cmd"]
    assert "hms_api.main" in captured["cmd"]
    # Verify remaining args are preserved
    assert "--port" in captured["cmd"]
    assert "9999" in captured["cmd"]

    # Verify env has the daemon child marker
    env = captured["kwargs"]["env"]
    assert env["_HMS_DAEMON_CHILD"] == "1"

    # Verify detach kwargs
    kwargs = captured["kwargs"]
    assert kwargs.get("start_new_session") is True


def test_daemonize_child_does_not_reexec(monkeypatch, tmp_path):
    """Child path: when _HMS_DAEMON_CHILD=1, daemonize() does NOT call
    Popen — it only redirects stdio."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("_HMS_DAEMON_CHILD", "1")

    log_path = tmp_path / "daemon.log"
    monkeypatch.setattr("hms_api.daemon.DAEMON_LOG_PATH", log_path)

    with (
        patch("hms_api.daemon.subprocess.Popen") as mock_popen,
        patch("hms_api.daemon._redirect_stdio_to_log") as mock_redirect,
    ):
        from hms_api.daemon import daemonize

        daemonize()
        mock_popen.assert_not_called()
        mock_redirect.assert_called_once()


def test_daemonize_windows_noop(monkeypatch, tmp_path):
    """On Windows, daemonize() just creates the log directory."""
    monkeypatch.setattr(sys, "platform", "win32")

    log_path = tmp_path / "subdir" / "daemon.log"
    monkeypatch.setattr("hms_api.daemon.DAEMON_LOG_PATH", log_path)

    with patch("hms_api.daemon.subprocess.Popen") as mock_popen:
        from hms_api.daemon import daemonize

        daemonize()
        mock_popen.assert_not_called()

    assert log_path.parent.exists()
