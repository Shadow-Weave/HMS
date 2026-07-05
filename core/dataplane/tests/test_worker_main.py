"""Tests for hms_api.worker.main entry-point helpers."""

import asyncio
import signal
from unittest.mock import MagicMock

from hms_api.worker.main import _install_shutdown_signal_handlers


def test_install_shutdown_signal_handlers_unix_path():
    """On platforms where asyncio supports signal handlers (Unix), both
    SIGINT and SIGTERM are registered and the helper reports success."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    handler = MagicMock()

    installed = _install_shutdown_signal_handlers(loop, handler)

    assert installed is True
    loop.add_signal_handler.assert_any_call(signal.SIGINT, handler)
    loop.add_signal_handler.assert_any_call(signal.SIGTERM, handler)
    assert loop.add_signal_handler.call_count == 2


def test_install_shutdown_signal_handlers_windows_path():
    """On Windows, asyncio's ProactorEventLoop raises NotImplementedError
    from add_signal_handler. The helper must swallow it and report failure
    so the worker keeps running with default Python signal behavior
    (regression test for issue #1411)."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.add_signal_handler.side_effect = NotImplementedError
    handler = MagicMock()

    installed = _install_shutdown_signal_handlers(loop, handler)

    assert installed is False
