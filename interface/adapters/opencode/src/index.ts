/**
 * HMS OpenCode Plugin — persistent long-term memory for OpenCode agents.
 *
 * Provides:
 *   - Custom tools: hms_retain, hms_recall, hms_reflect
 *   - Auto-retain on session.idle
 *   - Memory injection on session.created via system transform
 *   - Memory preservation during context compaction
 *
 * @example
 * ```json
 * // opencode.json
 * { "plugin": ["@hms-memory/opencode-hms"] }
 *
 * // With options:
 * { "plugin": [["@hms-memory/opencode-hms", { "bankId": "my-bank" }]] }
 * ```
 */

import type { Plugin } from "@opencode-ai/plugin";
import { HMSClient } from "@hms-memory/hms-client";
import { loadConfig } from "./config.js";
import { deriveBankId } from "./bank.js";
import { createTools } from "./tools.js";
import { createHooks, type PluginState } from "./hooks.js";
import { debugLog } from "./config.js";

// Module-level state persists across sessions (plugin is instantiated per session,
// but the module is loaded once per OpenCode server process).
const state: PluginState = {
  turnCount: 0,
  missionsSet: new Set(),
  recalledSessions: new Set(),
  lastRetainedTurn: new Map(),
};

const HMSPlugin: Plugin = async (input, options) => {
  const config = loadConfig(options);

  const apiUrl = config.hmsApiUrl;
  if (!apiUrl) {
    console.error(
      "[HMS] No API URL configured. Set HMS_API_URL environment variable " +
        "or add hmsApiUrl to ~/.hms/opencode.json"
    );
    // Return empty hooks — graceful degradation
    return {};
  }

  const client = new HMSClient({
    baseUrl: apiUrl,
    apiKey: config.hmsApiToken || undefined,
  });

  const bankId = deriveBankId(config, input.directory);
  debugLog(config, `Initialized with bank: ${bankId}, API: ${apiUrl}`);

  const tools = createTools(client, bankId, config, state.missionsSet);
  const hooks = createHooks(
    client,
    bankId,
    config,
    state,
    input.client as unknown as Parameters<typeof createHooks>[4]
  );

  return {
    tool: tools,
    ...hooks,
  };
};

// Named export for direct import
export { HMSPlugin };

// Default export is the Plugin function itself — OpenCode's loader calls the
// default export directly.
export default HMSPlugin;

// Re-export types for consumers
export type { HMSConfig } from "./config.js";
export type { PluginState } from "./hooks.js";
export { loadConfig } from "./config.js";
export { deriveBankId } from "./bank.js";
