"""Tests for EmbedManager interface."""

from unittest.mock import MagicMock, patch

from hms_embed import get_embed_manager
from hms_embed.daemon_embed_manager import DaemonEmbedManager


def test_sanitize_profile_name_via_db_url():
    """Test profile name sanitization through database URL generation."""
    manager = get_embed_manager()

    # Test None defaults to "default"
    assert manager.get_database_url(None) == "pg0://hms-embed-default"

    # Test simple alphanumeric names
    assert manager.get_database_url("myapp") == "pg0://hms-embed-myapp"
    assert manager.get_database_url("my-app") == "pg0://hms-embed-my-app"
    assert manager.get_database_url("my_app") == "pg0://hms-embed-my_app"
    assert manager.get_database_url("app123") == "pg0://hms-embed-app123"

    # Test special characters get replaced with dashes
    assert manager.get_database_url("my app") == "pg0://hms-embed-my-app"
    assert manager.get_database_url("my.app") == "pg0://hms-embed-my-app"
    assert manager.get_database_url("my@app!") == "pg0://hms-embed-my-app-"
    assert manager.get_database_url("My App 2.0!") == "pg0://hms-embed-My-App-2-0-"


def test_get_database_url_default():
    """Test database URL generation with default pg0."""
    manager = get_embed_manager()

    assert manager.get_database_url("myapp") == "pg0://hms-embed-myapp"
    assert manager.get_database_url("myapp", None) == "pg0://hms-embed-myapp"
    assert manager.get_database_url("myapp", "pg0") == "pg0://hms-embed-myapp"


def test_get_database_url_custom():
    """Test database URL generation with custom database."""
    manager = get_embed_manager()

    custom_url = "postgresql://user:pass@localhost/db"
    assert manager.get_database_url("myapp", custom_url) == custom_url
    assert manager.get_database_url("any-profile", custom_url) == custom_url


def test_manager_singleton():
    """Test that get_embed_manager returns functional instances."""
    manager1 = get_embed_manager()
    manager2 = get_embed_manager()

    # They should be independent instances but same type
    assert type(manager1) == type(manager2)

    # They should produce the same results
    assert manager1.get_database_url("test") == manager2.get_database_url("test")


def test_register_profile_skips_when_no_api_keys():
    """
    When config contains only short keys (no HMS_API_* prefix),
    _register_profile should not call create_profile, preserving any
    existing profile .env file.

    Regression test for https://github.com/hms-memory/hms/issues/894
    """
    manager = DaemonEmbedManager()
    manager._profile_manager = MagicMock()

    # Config with short keys (as passed from cli.py's get_config())
    config = {"llm_api_key": "sk-123", "llm_provider": "openai", "llm_model": "gpt-4o"}
    manager._register_profile("myprofile", 8100, config)

    manager._profile_manager.create_profile.assert_not_called()


def test_register_profile_calls_create_when_api_keys_present():
    """
    When config contains HMS_API_* keys, _register_profile should
    forward them to create_profile.
    """
    manager = DaemonEmbedManager()
    manager._profile_manager = MagicMock()

    config = {
        "HMS_API_LLM_PROVIDER": "openai",
        "HMS_API_LLM_API_KEY": "sk-123",
        "some_internal_key": "ignored",
    }
    manager._register_profile("myprofile", 8100, config)

    manager._profile_manager.create_profile.assert_called_once_with(
        "myprofile",
        8100,
        {"HMS_API_LLM_PROVIDER": "openai", "HMS_API_LLM_API_KEY": "sk-123"},
    )


def test_find_ui_command_uses_npx_yes_flag_for_published_control_plane(monkeypatch):
    """First-run UI installs must auto-confirm the published control-plane package."""
    manager = DaemonEmbedManager()
    monkeypatch.setenv("HMS_EMBED_CP_VERSION", "9.9.9")

    with patch("pathlib.Path.exists", return_value=False):
        assert manager._find_ui_command() == [
            "npx",
            "-y",
            "@hms-memory/hms-control-plane@9.9.9",
        ]


def test_find_api_command_prefers_installed_binary_over_uvx(tmp_path, monkeypatch):
    """
    When hms-api is installed alongside hms-embed (e.g. via
    `pip install hms-all`), _find_api_command should invoke that
    binary directly rather than shelling out to uvx. Uses sysconfig to
    locate the venv's scripts directory (issue #1401, #1240).
    """
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir()
    api_binary = scripts_dir / "hms-api"
    api_binary.touch()

    manager = DaemonEmbedManager()
    # Point __file__ away from monorepo so dev-mode check doesn't trigger
    monkeypatch.setattr("hms_embed.daemon_embed_manager.__file__", str(tmp_path / "hms_embed" / "daemon_embed_manager.py"))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.platform.system", lambda: "Linux")

    assert manager._find_api_command() == [str(api_binary)]


def test_find_api_command_target_install_uses_file_relative_fallback(tmp_path, monkeypatch):
    """
    When installed with `pip install --target`, sysconfig still points at the
    system/venv scripts dir (no binary there). The __file__-relative fallback
    should find the sibling binary in <target>/bin/ (issue #1240).
    """
    # sysconfig points to an empty venv scripts dir (no binary)
    venv_scripts = tmp_path / "venv_bin"
    venv_scripts.mkdir()

    # --target layout: binary sits next to site-packages contents
    target_dir = tmp_path / "target"
    pkg_dir = target_dir / "hms_embed"
    pkg_dir.mkdir(parents=True)
    fake_module = pkg_dir / "daemon_embed_manager.py"
    fake_module.write_text("")
    sibling_bin = target_dir / "bin" / "hms-api"
    sibling_bin.parent.mkdir()
    sibling_bin.touch()

    manager = DaemonEmbedManager()
    monkeypatch.setattr("hms_embed.daemon_embed_manager.__file__", str(fake_module))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(venv_scripts))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.platform.system", lambda: "Linux")

    assert manager._find_api_command() == [str(sibling_bin)]


def test_find_api_command_falls_back_to_uvx_when_no_binary(tmp_path, monkeypatch):
    """Without an installed binary or dev checkout, fall back to uvx."""
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir()
    # No hms-api binary in scripts_dir

    manager = DaemonEmbedManager()
    monkeypatch.setattr("hms_embed.daemon_embed_manager.__file__", str(tmp_path / "hms_embed" / "daemon_embed_manager.py"))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.platform.system", lambda: "Linux")
    monkeypatch.setenv("HMS_EMBED_API_VERSION", "1.2.3")

    assert manager._find_api_command() == ["uvx", "hms-api@1.2.3"]


def test_find_api_command_windows_uses_exe_suffix(tmp_path, monkeypatch):
    """On Windows, the installed binary has a .exe suffix."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    api_binary = scripts_dir / "hms-api.exe"
    api_binary.touch()

    manager = DaemonEmbedManager()
    # Point __file__ away from monorepo so dev-mode check doesn't trigger
    monkeypatch.setattr("hms_embed.daemon_embed_manager.__file__", str(tmp_path / "hms_embed" / "daemon_embed_manager.py"))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hms_embed.daemon_embed_manager.platform.system", lambda: "Windows")

    assert manager._find_api_command() == [str(api_binary)]
