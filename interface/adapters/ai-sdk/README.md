# HMS Memory Integration for Vercel AI SDK

Give your AI agents persistent, human-like memory using [HMS](https://docs.hms.local) with the [Vercel AI SDK](https://ai-sdk.dev).

## Quick Start

```bash
npm install @hms-memory/hms-ai-sdk @hms-memory/hms-client ai zod
```

```typescript
import { HMSClient } from "@hms-memory/hms-client";
import { createHMSTools } from "@hms-memory/hms-ai-sdk";
import { generateText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";

// 1. Initialize HMS client
const hmsClient = new HMSClient({
  apiUrl: "http://localhost:8000",
});

// 2. Create memory tools
const tools = createHMSTools({ client: hmsClient });

// 3. Use with AI SDK
const result = await generateText({
  model: anthropic("claude-sonnet-4-20250514"),
  tools,
  system: `You have long-term memory. Use:
  - 'recall' to search past conversations
  - 'retain' to remember important information
  - 'reflect' to synthesize insights from memories`,
  prompt: "Remember that Alice loves hiking and prefers spicy food",
});

console.log(result.text);
```

## Features

✅ **Three Memory Tools**: `retain` (store), `recall` (retrieve), and `reflect` (reason over memories)
✅ **AI SDK 6 Native**: Works with `generateText`, `streamText`, and `ToolLoopAgent`
✅ **Multi-User Support**: Dynamic bank IDs per call for multi-user scenarios
✅ **Type-Safe**: Full TypeScript support with Zod schemas
✅ **Flexible Client**: Works with the official TypeScript client or custom HTTP clients

## Documentation

📖 **[Full Documentation](https://docs.hms.local/sdks/integrations/ai-sdk)**

The complete documentation includes:

- Detailed tool descriptions and parameters
- Advanced usage patterns (streaming, multi-user, ToolLoopAgent)
- HTTP client example (no dependencies)
- TypeScript types and API reference
- Best practices and system prompt examples

## Running HMS Locally

```bash
# Install and run with embedded mode (no setup required)
uvx hms-embed@latest -p myapp daemon start

# The API will be available at http://localhost:8000
```

## Examples

Full examples are available in the [GitHub repository](https://github.com/hms-memory/hms/tree/main/examples/ai-sdk).

## Support

- [Documentation](https://docs.hms.local)
- [GitHub Issues](https://github.com/hms-memory/hms/issues)
- Email: support@hms.local

## License

MIT
