# hms-nemoclaw

One-command setup for [HMS](https://docs.hms.local) persistent memory on [NemoClaw](https://nemoclaw.ai) sandboxes.

NemoClaw runs [OpenClaw](https://openclaw.ai) inside an OpenShell sandbox with strict network egress policies. This package automates the full setup: installing the `hms-openclaw` plugin, configuring external API mode, merging the HMS egress rule into your sandbox policy, and restarting the gateway.

## Quick Start

```bash
npx @hms-memory/hms-nemoclaw setup \
  --sandbox my-assistant \
  --api-url https://api.hms.local \
  --api-token <your-api-key> \
  --bank-prefix my-sandbox
```

Get an API key at [HMS Cloud](https://ui.hms.local/signup).

## Documentation

Full setup guide, pitfalls, and troubleshooting:

**[NemoClaw Integration Documentation](https://docs.hms.local/sdks/integrations/nemoclaw)**

Or see [NEMOCLAW.md](./NEMOCLAW.md) in this directory for a step-by-step walkthrough.

## CLI Reference

```
hms-nemoclaw setup [options]

Options:
  --sandbox <name>       NemoClaw sandbox name (required)
  --api-url <url>        HMS API URL (required)
  --api-token <token>    HMS API token (required)
  --bank-prefix <prefix> Memory bank prefix (default: "nemoclaw")
  --skip-policy          Skip sandbox network policy update
  --skip-plugin-install  Skip openclaw plugin installation
  --dry-run              Preview changes without applying
  --help                 Show help
```

## What It Does

1. **Preflight** — verifies `openshell` and `openclaw` are installed
2. **Install plugin** — runs `openclaw plugins install @hms-memory/hms-openclaw`
3. **Configure plugin** — writes external API mode config to `~/.openclaw/openclaw.json`
4. **Apply policy** — reads current sandbox policy, merges HMS egress rule, re-applies via `openshell policy set`
5. **Restart gateway** — runs `openclaw gateway restart`

## Links

- [HMS Documentation](https://docs.hms.local)
- [NemoClaw](https://nemoclaw.ai)
- [OpenClaw](https://openclaw.ai)
- [GitHub Repository](https://github.com/hms-memory/hms)

## License

MIT
