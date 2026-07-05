"""Tests for daemon mode host/port resolution (resolve_daemon_host_port).

Verifies that --daemon honors explicit --host / HMS_API_HOST overrides
instead of unconditionally pinning to 127.0.0.1.

See: https://github.com/hms-memory/hms/issues/1402
"""

from hms_api.daemon import DEFAULT_DAEMON_PORT
from hms_api.main import resolve_daemon_host_port

# Default config values matching production defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888


class TestResolveDaemonHostPort:
    """Test resolve_daemon_host_port under various override scenarios."""

    def test_defaults_to_localhost_when_no_override(self, monkeypatch):
        """With no explicit host setting, daemon should bind to 127.0.0.1."""
        monkeypatch.delenv("HMS_API_HOST", raising=False)
        host, port = resolve_daemon_host_port(
            args_host=DEFAULT_HOST, args_port=DEFAULT_PORT,
            config_host=DEFAULT_HOST, config_port=DEFAULT_PORT,
        )
        assert host == "127.0.0.1"
        assert port == DEFAULT_DAEMON_PORT

    def test_honors_cli_host_flag(self, monkeypatch):
        """--host 0.0.0.0 --daemon should bind to 0.0.0.0, not 127.0.0.1."""
        monkeypatch.delenv("HMS_API_HOST", raising=False)
        host, port = resolve_daemon_host_port(
            args_host="0.0.0.0", args_port=DEFAULT_PORT,
            config_host="127.0.0.1", config_port=DEFAULT_PORT,
        )
        assert host == "0.0.0.0"

    def test_honors_env_var_host(self, monkeypatch):
        """HMS_API_HOST=0.0.0.0 --daemon should bind to 0.0.0.0."""
        monkeypatch.setenv("HMS_API_HOST", "0.0.0.0")
        # When env var is set, config.host already reflects it, so
        # args_host == config_host — but the env var presence is the signal.
        host, port = resolve_daemon_host_port(
            args_host="0.0.0.0", args_port=DEFAULT_PORT,
            config_host="0.0.0.0", config_port=DEFAULT_PORT,
        )
        assert host == "0.0.0.0"

    def test_honors_custom_host_via_env(self, monkeypatch):
        """HMS_API_HOST=10.0.0.5 should be respected in daemon mode."""
        monkeypatch.setenv("HMS_API_HOST", "10.0.0.5")
        host, _ = resolve_daemon_host_port(
            args_host="10.0.0.5", args_port=DEFAULT_PORT,
            config_host="10.0.0.5", config_port=DEFAULT_PORT,
        )
        assert host == "10.0.0.5"

    def test_cli_flag_overrides_env_var(self, monkeypatch):
        """--host flag should take precedence over env var."""
        monkeypatch.setenv("HMS_API_HOST", "10.0.0.5")
        host, _ = resolve_daemon_host_port(
            args_host="192.168.1.1", args_port=DEFAULT_PORT,
            config_host="10.0.0.5", config_port=DEFAULT_PORT,
        )
        assert host == "192.168.1.1"

    def test_custom_port_preserved(self, monkeypatch):
        """--port 9999 --daemon should keep 9999, not switch to daemon default."""
        monkeypatch.delenv("HMS_API_HOST", raising=False)
        _, port = resolve_daemon_host_port(
            args_host=DEFAULT_HOST, args_port=9999,
            config_host=DEFAULT_HOST, config_port=DEFAULT_PORT,
        )
        assert port == 9999

    def test_default_port_becomes_daemon_port(self, monkeypatch):
        """No --port flag in daemon mode should use DEFAULT_DAEMON_PORT."""
        monkeypatch.delenv("HMS_API_HOST", raising=False)
        _, port = resolve_daemon_host_port(
            args_host=DEFAULT_HOST, args_port=DEFAULT_PORT,
            config_host=DEFAULT_HOST, config_port=DEFAULT_PORT,
        )
        assert port == DEFAULT_DAEMON_PORT
