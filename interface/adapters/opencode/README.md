# @hms-memory/opencode-hms

HMS memory plugin for [OpenCode](https://opencode.ai) — give your AI coding agent persistent long-term memory across sessions.

## Features

- **Custom tools**: `hms_retain`, `hms_recall`, `hms_reflect` — the agent calls these explicitly
- **Auto-retain**: Captures conversation on `session.idle` and stores to HMS
- **Memory injection**: Recalls relevant memories when a new session starts
- **Compaction hook**: Injects memories during context compaction so they survive window trimming

## Quick Start

### 1. Enable the plugin

Add to your `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@hms-memory/opencode-hms"]
}
```

OpenCode auto-installs plugins listed here on startup — no `npm install` required.

### 2. Point to your HMS server

```bash
# Self-hosted
export HMS_API_URL="http://localhost:8888"

# Optional: override the memory bank ID
export HMS_BANK_ID="my-project"
```

### Using HMS Cloud

Get an API key at [ui.hms.local/connect](https://ui.hms.local/connect), then either export env vars:

```bash
export HMS_API_URL="https://api.hms.local"
export HMS_API_TOKEN="your-api-key"
```

Or configure inline in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    [
      "@hms-memory/opencode-hms",
      {
        "hmsApiUrl": "https://api.hms.local",
        "hmsApiToken": "your-api-key"
      }
    ]
  ]
}
```

## Configuration

### Plugin Options

Pass options directly in `opencode.json`:

```json
{
  "plugin": [
    [
      "@hms-memory/opencode-hms",
      {
        "hmsApiUrl": "http://localhost:8888",
        "bankId": "my-project",
        "autoRecall": true,
        "autoRetain": true,
        "recallBudget": "mid"
      }
    ]
  ]
}
```

### Config File

Create `~/.hms/opencode.json` for persistent configuration:

```json
{
  "hmsApiUrl": "http://localhost:8888",
  "hmsApiToken": "your-api-key",
  "recallBudget": "mid",
  "retainEveryNTurns": 3,
  "debug": false
}
```

### Environment Variables

| Variable                      | Description                         | Default        |
| ----------------------------- | ----------------------------------- | -------------- |
| `HMS_API_URL`           | HMS API base URL              | (required)     |
| `HMS_API_TOKEN`         | API key for authentication          | (none)         |
| `HMS_BANK_ID`           | Static memory bank ID               | `opencode`     |
| `HMS_AGENT_NAME`        | Agent name for dynamic bank IDs     | `opencode`     |
| `HMS_AUTO_RECALL`       | Auto-recall on session start        | `true`         |
| `HMS_AUTO_RETAIN`       | Auto-retain on session idle         | `true`         |
| `HMS_RETAIN_MODE`       | `full-session` or `last-turn`       | `full-session` |
| `HMS_RECALL_BUDGET`     | Recall budget: `low`, `mid`, `high` | `mid`          |
| `HMS_RECALL_MAX_TOKENS` | Max tokens for recall results       | `1024`         |
| `HMS_DYNAMIC_BANK_ID`   | Enable dynamic bank ID derivation   | `false`        |
| `HMS_BANK_MISSION`      | Bank mission/context                | (none)         |
| `HMS_DEBUG`             | Enable debug logging                | `false`        |

### Configuration Priority

Settings are loaded in this order (later wins):

1. Built-in defaults
2. `~/.hms/opencode.json`
3. Plugin options from `opencode.json`
4. Environment variables

## Tools

### `hms_retain`

Store information in long-term memory. The agent uses this to save important facts, user preferences, project context, and decisions.

### `hms_recall`

Search long-term memory. The agent uses this proactively before answering questions where prior context would help.

### `hms_reflect`

Generate a synthesized answer from long-term memory. Unlike recall (raw memories), reflect produces a coherent summary.

## Dynamic Bank IDs

For multi-project setups, enable dynamic bank ID derivation:

```bash
export HMS_DYNAMIC_BANK_ID=true
```

The bank ID is composed from granularity fields (default: `agent::project`). Supported fields: `agent`, `project`, `gitProject`, `channel`, `user`.

- `project` uses the working directory basename. With this field, separate git worktrees of the same repository end up with different bank IDs because their paths differ.
- `gitProject` resolves to the main worktree's basename via `git rev-parse --git-common-dir`, so all linked worktrees of the same repository share a single bank. Falls back to the working directory basename when git is unavailable or the directory is not a repo. Use this in place of `project` if you want worktrees to share memory:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "gitProject"]
}
```

**Note:** The bank ID is derived once when the plugin loads, from environment variables set before OpenCode starts. These dimensions are process-scoped — they don't change per session within a running OpenCode process. For per-user isolation, set the env vars before launching each user's OpenCode instance:

```bash
export HMS_CHANNEL_ID="slack-general"
export HMS_USER_ID="user123"
```

## Development

```bash
npm install
npm test        # Run tests
npm run build   # Build to dist/
```

## License

MIT
