---
name: hms:setup
description: Register hms-memory hooks into Claude Code settings. Run this once after installing the plugin.
---

Register the hms-memory hooks into `~/.claude/settings.json` by running the setup script:

```bash
python3 "$CLAUDE_PLUGIN_ROOT/scripts/setup_hooks.py"
```

If `CLAUDE_PLUGIN_ROOT` is not set, find the path manually:

```bash
ls ~/.claude/plugins/cache/hms/hms-memory/
```

Then run:

```bash
python3 ~/.claude/plugins/cache/hms/hms-memory/<version>/scripts/setup_hooks.py
```

After the script completes, restart Claude Code for the hooks to take effect. You should see `[HMS]` log lines on the next session start.
