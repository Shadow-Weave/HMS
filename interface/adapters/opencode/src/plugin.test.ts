import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the HMSClient before importing the plugin
vi.mock("@hms-memory/hms-client", () => {
  const MockHMSClient = vi.fn(function (this: any) {
    this.retain = vi.fn().mockResolvedValue({});
    this.recall = vi.fn().mockResolvedValue({ results: [] });
    this.reflect = vi.fn().mockResolvedValue({ text: "" });
    this.createBank = vi.fn().mockResolvedValue({});
  });
  return { HMSClient: MockHMSClient };
});

import { HMSPlugin } from "./index.js";
import { HMSClient } from "@hms-memory/hms-client";

const mockPluginInput = {
  client: {
    session: {
      messages: vi.fn().mockResolvedValue({ data: [] }),
    },
  },
  project: { id: "test-project", worktree: "/tmp/test", vcs: "git" },
  directory: "/tmp/test-project",
  worktree: "/tmp/test-project",
  serverUrl: new URL("http://localhost:3000"),
  $: {} as any,
};

describe("HMSPlugin", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("HMS_")) delete process.env[key];
    }
    vi.clearAllMocks();
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("returns empty hooks when no API URL configured", async () => {
    const result = await HMSPlugin(mockPluginInput as any);
    expect(result).toEqual({});
    expect(HMSClient).not.toHaveBeenCalled();
  });

  it("returns tools and hooks when configured", async () => {
    process.env.HMS_API_URL = "http://localhost:8888";

    const result = await HMSPlugin(mockPluginInput as any);

    expect(HMSClient).toHaveBeenCalledWith({
      baseUrl: "http://localhost:8888",
      apiKey: undefined,
    });

    expect(result.tool).toBeDefined();
    expect(result.tool!.hms_retain).toBeDefined();
    expect(result.tool!.hms_recall).toBeDefined();
    expect(result.tool!.hms_reflect).toBeDefined();
    expect(result.event).toBeDefined();
    expect(result["experimental.session.compacting"]).toBeDefined();
    expect(result["experimental.chat.system.transform"]).toBeDefined();
  });

  it("passes API key when configured", async () => {
    process.env.HMS_API_URL = "http://localhost:8888";
    process.env.HMS_API_TOKEN = "my-token";

    await HMSPlugin(mockPluginInput as any);

    expect(HMSClient).toHaveBeenCalledWith({
      baseUrl: "http://localhost:8888",
      apiKey: "my-token",
    });
  });

  it("accepts plugin options", async () => {
    const result = await HMSPlugin(mockPluginInput as any, {
      hmsApiUrl: "http://example.com",
      bankId: "custom-bank",
    });

    expect(result.tool).toBeDefined();
    expect(HMSClient).toHaveBeenCalledWith({
      baseUrl: "http://example.com",
      apiKey: undefined,
    });
  });
});

describe("HMSPlugin state sharing", () => {
  beforeEach(() => {
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("HMS_")) delete process.env[key];
    }
    vi.clearAllMocks();
  });

  it("shares state across multiple plugin instantiations (sessions)", async () => {
    process.env.HMS_API_URL = "http://localhost:8888";

    // Simulate two sessions calling the plugin (OpenCode instantiates per session)
    const result1 = await HMSPlugin(mockPluginInput as any);
    const result2 = await HMSPlugin(mockPluginInput as any);

    // Trigger session.created on session 1 — should track 'sess-A'
    await result1.event!({
      event: { type: "session.created", properties: { info: { id: "sess-A" } } },
    });

    // Session 2's system transform should see 'sess-A' because state is shared
    const output = { system: [] as string[] };
    await result2["experimental.chat.system.transform"]!(
      { sessionID: "sess-A", model: {} },
      output
    );

    // The recall was attempted (state was shared — sess-A was found in recalledSessions).
    // If state were per-instance, result2 would have an empty recalledSessions and skip recall.
    // result2 uses the second HMSClient instance (index 1).
    const clientInstance = (HMSClient as any).mock.instances[1];
    expect(clientInstance.recall).toHaveBeenCalled();
  });
});

describe("plugin default export", () => {
  it("default-exports the Plugin function itself", async () => {
    const mod = await import("./index.js");
    expect(typeof mod.default).toBe("function");
    // OpenCode iterates Object.entries(mod) and calls every export as a
    // Plugin factory, deduping by reference. The default export must be
    // the same reference as the named HMSPlugin export to avoid
    // running the factory twice.
    expect(mod.default).toBe(mod.HMSPlugin);
  });
});
