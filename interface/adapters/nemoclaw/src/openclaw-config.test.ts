import { describe, it, expect } from "vitest";
import { mergePluginConfig } from "./openclaw-config.js";
import type { OpenClawConfig, HMSPluginConfig } from "./openclaw-config.js";

const PLUGIN_CONFIG: HMSPluginConfig = {
  hmsApiUrl: "https://api.hms.local",
  hmsApiToken: "hsk_test123",
  llmProvider: "claude-code",
  dynamicBankId: false,
  bankIdPrefix: "my-sandbox",
};

const BASE_CONFIG: OpenClawConfig = {
  meta: { lastTouchedVersion: "2026.3.2" },
  gateway: { port: 18789, mode: "local" },
  agents: {
    defaults: { model: { primary: "openai/gpt-5" } },
  },
  plugins: {
    slots: { memory: "memory-core" },
    entries: {
      "memory-core": { enabled: false },
    },
  },
};

describe("mergePluginConfig", () => {
  it("sets hms-openclaw as the memory slot", () => {
    const result = mergePluginConfig(BASE_CONFIG, PLUGIN_CONFIG);
    expect(result.plugins?.slots?.memory).toBe("hms-openclaw");
  });

  it("enables the hms-openclaw entry", () => {
    const result = mergePluginConfig(BASE_CONFIG, PLUGIN_CONFIG);
    expect(result.plugins?.entries?.["hms-openclaw"]?.enabled).toBe(true);
  });

  it("writes the full plugin config", () => {
    const result = mergePluginConfig(BASE_CONFIG, PLUGIN_CONFIG);
    const config = result.plugins?.entries?.["hms-openclaw"]?.config;
    expect(config?.hmsApiUrl).toBe("https://api.hms.local");
    expect(config?.hmsApiToken).toBe("hsk_test123");
    expect(config?.llmProvider).toBe("claude-code");
    expect(config?.dynamicBankId).toBe(false);
    expect(config?.bankIdPrefix).toBe("my-sandbox");
  });

  it("preserves existing top-level config fields", () => {
    const result = mergePluginConfig(BASE_CONFIG, PLUGIN_CONFIG);
    expect(result.gateway).toEqual({ port: 18789, mode: "local" });
    expect(result.agents).toBeDefined();
  });

  it("preserves existing plugin entries", () => {
    const result = mergePluginConfig(BASE_CONFIG, PLUGIN_CONFIG);
    expect(result.plugins?.entries?.["memory-core"]?.enabled).toBe(false);
  });

  it("merges into existing hms-openclaw entry without overwriting other fields", () => {
    const configWithExisting: OpenClawConfig = {
      ...BASE_CONFIG,
      plugins: {
        ...BASE_CONFIG.plugins,
        entries: {
          ...BASE_CONFIG.plugins?.entries,
          "hms-openclaw": {
            enabled: true,
            config: { embedPackagePath: "/some/local/path" },
          },
        },
      },
    };
    const result = mergePluginConfig(configWithExisting, PLUGIN_CONFIG);
    const config = result.plugins?.entries?.["hms-openclaw"]?.config;
    // New fields written
    expect(config?.hmsApiUrl).toBe("https://api.hms.local");
    // Existing custom field preserved
    expect(config?.embedPackagePath).toBe("/some/local/path");
  });

  it("handles missing plugins section gracefully", () => {
    const minimal: OpenClawConfig = { gateway: { port: 18789 } };
    const result = mergePluginConfig(minimal, PLUGIN_CONFIG);
    expect(result.plugins?.slots?.memory).toBe("hms-openclaw");
    expect(result.plugins?.entries?.["hms-openclaw"]?.enabled).toBe(true);
  });

  it("does not mutate the original config", () => {
    const original = JSON.parse(JSON.stringify(BASE_CONFIG)) as OpenClawConfig;
    mergePluginConfig(BASE_CONFIG, PLUGIN_CONFIG);
    expect(JSON.stringify(BASE_CONFIG)).toBe(JSON.stringify(original));
  });

  it("records install metadata", () => {
    const result = mergePluginConfig(BASE_CONFIG, PLUGIN_CONFIG);
    const install = result.plugins?.installs?.["hms-openclaw"] as Record<string, unknown>;
    expect(install?.source).toBe("npm");
    expect(install?.version).toBe("latest");
    expect(typeof install?.installedAt).toBe("string");
  });
});
