# Using hms-openclaw with NemoClaw

This guide covers running the `hms-openclaw` plugin inside a [NemoClaw](https://nemoclaw.ai) sandbox. NemoClaw runs OpenClaw inside an OpenShell sandbox, so the plugin's outbound calls to `api.hms.local` must be explicitly allowed in the sandbox's network egress policy.

## Prerequisites

- NemoClaw installed and a sandbox created (`nemoclaw onboard`)
- OpenClaw installed (`brew install openclaw` or equivalent)
- A HMS API key from [ui.hms.local](https://ui.hms.local)
- The plugin source built (`npm run build` in this directory)

## Step 1: Create a HMS memory bank

Create the bank the plugin will write to. The bank ID follows the pattern `{bankIdPrefix}-openclaw` when `dynamicBankId` is false:

```bash
curl -X PUT "https://api.hms.local/v1/default/banks/my-sandbox-openclaw" \
  -H "Authorization: Bearer <your-hms-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"mission": "Memory bank for my NemoClaw sandbox."}'
```

## Step 2: Install the plugin

Install the plugin as a copy (not a symlink) so the OpenClaw LaunchAgent can access it:

```bash
# Build first if you haven't already
npm run build

# Install (copy, not link — required for LaunchAgent access)
openclaw plugins install /path/to/interface/adapters/openclaw
```

Alternatively, install from npm:

```bash
openclaw plugins install @hms-memory/hms-openclaw
```

## Step 3: Configure the plugin

Add the plugin config to `~/.openclaw/openclaw.json` under `plugins.entries.hms-openclaw`:

```json
{
  "plugins": {
    "entries": {
      "hms-openclaw": {
        "enabled": true,
        "config": {
          "hmsApiUrl": "https://api.hms.local",
          "hmsApiToken": "<your-hms-api-key>",
          "llmProvider": "claude-code",
          "dynamicBankId": false,
          "bankIdPrefix": "my-sandbox"
        }
      }
    }
  }
}
```

**Config notes:**

| Field                                   | Value                  | Why                                                                                                                       |
| --------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `hmsApiUrl` + `hmsApiToken` | External API URL + key | Skips the local daemon; no `uvx`/`uv` required inside the sandbox                                                         |
| `llmProvider: "claude-code"`            | `"claude-code"`        | Satisfies LLM detection without a separate API key — Claude Code is available in the sandbox via the `claude_code` policy |
| `dynamicBankId: false`                  | `false`                | All conversations write to one bank; easier to verify during testing                                                      |
| `bankIdPrefix`                          | e.g. `"my-sandbox"`    | Results in bank ID `my-sandbox-openclaw`                                                                                  |

> **Note:** The gateway log will say `Dynamic bank IDs disabled - using static bank: openclaw` — this is a misleading log message. The actual bank ID used at runtime correctly applies the prefix (e.g. `my-sandbox-openclaw`). You can verify by watching for `[HMS] Default bank: my-sandbox-openclaw` in the logs after full initialization.

## Step 4: Add the HMS network policy to the sandbox

The sandbox blocks all outbound traffic by default. You need to add `api.hms.local` to the egress policy.

Get the current full policy by running `openshell sandbox get <name>` and save it to a YAML file, then add the `hms` block under `network_policies`:

```yaml
network_policies:
  # ... your existing policies ...
  hms:
    name: hms
    endpoints:
      - host: api.hms.local
        port: 443
        protocol: rest
        tls: terminate
        enforcement: enforce
        rules:
          - allow:
              method: GET
              path: /**
          - allow:
              method: POST
              path: /**
          - allow:
              method: PUT
              path: /**
    binaries:
      - path: /usr/local/bin/openclaw
```

Apply it:

```bash
openshell policy set <sandbox-name> --policy /path/to/full-policy.yaml --wait
```

> **Important:** `openshell policy set` replaces the entire policy, not just patches it. Make sure your YAML includes all existing network policies or they will be removed.

Verify the policy loaded:

```bash
openshell policy get <sandbox-name>
# Should show: Status: Loaded, and version incremented
```

## Step 5: Restart the OpenClaw gateway

```bash
openclaw gateway restart
```

Watch the logs to confirm the plugin loaded and the API is reachable:

```bash
# Should see:
# [HMS] Plugin loaded successfully
# [HMS] ✓ Using external API: https://api.hms.local
# [HMS] External API health: {"status":"healthy","database":"connected"}
# [HMS] Default bank: my-sandbox-openclaw
# [HMS] ✓ Ready (external API mode)
grep HMS ~/.openclaw/logs/gateway.log | tail -20
```

## Step 6: Test

Send a message to the agent:

```bash
openclaw agent --agent main --session-id test-1 \
  -m "My name is Ben and I work on HMS. I prefer detailed commit messages."
```

Verify memory was retained (check logs):

```bash
grep "Retained\|agent_end" ~/.openclaw/logs/gateway.log | tail -5
# Should see: [HMS] Retained N messages to bank my-sandbox-openclaw for session ...
```

Test recall in a new session:

```bash
openclaw agent --agent main --session-id test-2 \
  -m "What do you remember about me?"
# Should recall your name and preferences from the previous session
```

You can also verify directly against the API:

```bash
curl -s -X POST "https://api.hms.local/v1/default/banks/my-sandbox-openclaw/memories/recall" \
  -H "Authorization: Bearer <your-hms-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "what do you know about the user", "max_tokens": 512}'
```

## Troubleshooting

**Plugin fails to load with `EPERM: operation not permitted, scandir`**

You used `--link` when installing. The OpenClaw LaunchAgent runs under a restricted macOS security context and cannot access `~/Documents` or other user directories by symlink. Reinstall without `--link`:

```bash
openclaw plugins uninstall hms-openclaw
openclaw plugins install /path/to/interface/adapters/openclaw  # no --link
```

**`[HMS] Failed to retain memory (HTTP 403)`**

The sandbox network policy is blocking the outbound call. Check that:

1. The `hms` network policy block is present in your policy YAML
2. The policy was applied and shows `Status: Loaded` (`openshell policy get <name>`)
3. The `binaries` list includes `/usr/local/bin/openclaw`

**Gateway restart times out but then recovers**

This is normal on first restart after installing a plugin — the LaunchAgent takes a moment to reload. The gateway is healthy if `openclaw gateway status` shows `RPC probe: ok`.

**`openclaw agent` fails with `Pass --to, --session-id, or --agent`**

You need to specify a session. Use `--agent main` to use the default agent, or `--session-id <any-string>` to create a named session.
