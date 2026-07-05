#!/usr/bin/env python3
"""SessionStart hook: health check and daemon pre-start.

Fires once when a Codex session begins. Verifies the HMS server is
reachable, and kicks off a background daemon pre-start if not — so it's
ready by the first recall or retain hook.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.client import HMSClient
from lib.config import debug_log, load_config
from lib.daemon import get_api_url, prestart_daemon_background


def main():
    config = load_config()

    if not config.get("autoRecall") and not config.get("autoRetain"):
        debug_log(config, "Both autoRecall and autoRetain disabled, skipping session start")
        return

    # Consume stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    debug_log(config, f"SessionStart hook, session: {hook_input.get('session_id', 'unknown')}")

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
        HMSClient(api_url, config.get("hmsApiToken"))
        debug_log(config, f"HMS server reachable at {api_url}")
    except (RuntimeError, ValueError) as e:
        debug_log(config, f"HMS not running, initiating background pre-start: {e}")
        prestart_daemon_background(config, debug_fn=_dbg)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[HMS] SessionStart error: {e}", file=sys.stderr)
        sys.exit(0)
