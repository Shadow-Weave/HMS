import { describe, it, expect } from "vitest";
import { hasHMSPolicy, mergeHMSPolicy, serializePolicy } from "./policy-writer.js";
import { parseSandboxPolicy } from "./policy-reader.js";
import type { SandboxPolicy } from "./types.js";
import { HMS_HOST, OPENCLAW_BINARY } from "./types.js";

const BASE_POLICY: SandboxPolicy = {
  version: 1,
  filesystem_policy: {
    include_workdir: true,
    read_only: ["/usr", "/lib"],
    read_write: ["/sandbox", "/tmp"],
  },
  network_policies: {
    claude_code: {
      name: "claude_code",
      endpoints: [
        {
          host: "api.anthropic.com",
          port: 443,
          rules: [{ allow: { method: "*", path: "/**" } }],
        },
      ],
      binaries: [{ path: "/usr/local/bin/claude" }],
    },
  },
};

describe("hasHMSPolicy", () => {
  it("returns false when no hms policy exists", () => {
    expect(hasHMSPolicy(BASE_POLICY)).toBe(false);
  });

  it("returns false when network_policies is undefined", () => {
    expect(hasHMSPolicy({ version: 1 })).toBe(false);
  });

  it("returns true when hms policy is present", () => {
    const withHMS = mergeHMSPolicy(BASE_POLICY);
    expect(hasHMSPolicy(withHMS)).toBe(true);
  });
});

describe("mergeHMSPolicy", () => {
  it("adds the hms network policy block", () => {
    const result = mergeHMSPolicy(BASE_POLICY);
    expect(result.network_policies?.hms).toBeDefined();
    expect(result.network_policies?.hms?.endpoints[0].host).toBe(HMS_HOST);
  });

  it("preserves all existing network policies", () => {
    const result = mergeHMSPolicy(BASE_POLICY);
    expect(result.network_policies?.claude_code).toBeDefined();
    expect(result.network_policies?.claude_code?.name).toBe("claude_code");
  });

  it("sets the correct binary path", () => {
    const result = mergeHMSPolicy(BASE_POLICY);
    const binaries = result.network_policies?.hms?.binaries ?? [];
    expect(binaries.some((b) => b.path === OPENCLAW_BINARY)).toBe(true);
  });

  it("includes GET, POST, and PUT rules", () => {
    const result = mergeHMSPolicy(BASE_POLICY);
    const rules = result.network_policies?.hms?.endpoints[0].rules ?? [];
    const methods = rules.map((r) => r.allow.method);
    expect(methods).toContain("GET");
    expect(methods).toContain("POST");
    expect(methods).toContain("PUT");
  });

  it("is idempotent — merging twice yields the same result", () => {
    const once = mergeHMSPolicy(BASE_POLICY);
    const twice = mergeHMSPolicy(once);
    expect(JSON.stringify(twice.network_policies?.hms)).toBe(
      JSON.stringify(once.network_policies?.hms)
    );
  });

  it("does not mutate the original policy", () => {
    const original = JSON.parse(JSON.stringify(BASE_POLICY)) as SandboxPolicy;
    mergeHMSPolicy(BASE_POLICY);
    expect(BASE_POLICY.network_policies?.hms).toBeUndefined();
    expect(JSON.stringify(BASE_POLICY)).toBe(JSON.stringify(original));
  });
});

describe("serializePolicy", () => {
  it("produces valid YAML that round-trips through parseSandboxPolicy", () => {
    const merged = mergeHMSPolicy(BASE_POLICY);
    const yamlStr = serializePolicy(merged);
    // Wrap in Policy: header as parseSandboxPolicy expects
    const wrapped =
      "Policy:\n" +
      yamlStr
        .split("\n")
        .map((l) => `  ${l}`)
        .join("\n");
    const reparsed = parseSandboxPolicy(wrapped);
    expect(reparsed.version).toBe(merged.version);
    expect(reparsed.network_policies?.hms?.endpoints[0].host).toBe(HMS_HOST);
    expect(reparsed.network_policies?.claude_code).toBeDefined();
  });

  it("includes all network policies in output", () => {
    const merged = mergeHMSPolicy(BASE_POLICY);
    const yaml = serializePolicy(merged);
    expect(yaml).toContain("claude_code:");
    expect(yaml).toContain("hms:");
    expect(yaml).toContain(HMS_HOST);
  });
});
