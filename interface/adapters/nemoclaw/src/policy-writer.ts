import yaml from "js-yaml";
import type { SandboxPolicy } from "./types.js";
import { HMS_POLICY_NAME, HMS_HOST, OPENCLAW_BINARY } from "./types.js";

const HMS_NETWORK_POLICY = {
  name: HMS_POLICY_NAME,
  endpoints: [
    {
      host: HMS_HOST,
      port: 443,
      protocol: "rest",
      tls: "terminate",
      enforcement: "enforce",
      rules: [
        { allow: { method: "GET", path: "/**" } },
        { allow: { method: "POST", path: "/**" } },
        { allow: { method: "PUT", path: "/**" } },
      ],
    },
  ],
  binaries: [{ path: OPENCLAW_BINARY }],
};

/**
 * Returns true if the policy already has a correct HMS network policy entry.
 */
export function hasHMSPolicy(policy: SandboxPolicy): boolean {
  const np = policy.network_policies?.[HMS_POLICY_NAME];
  if (!np) return false;
  return np.endpoints?.some((e) => e.host === HMS_HOST) ?? false;
}

/**
 * Merge the HMS network policy block into a SandboxPolicy.
 * Idempotent — if the block already exists and is correct, returns policy unchanged.
 */
export function mergeHMSPolicy(policy: SandboxPolicy): SandboxPolicy {
  const updated: SandboxPolicy = {
    ...policy,
    network_policies: {
      ...(policy.network_policies ?? {}),
      [HMS_POLICY_NAME]: HMS_NETWORK_POLICY,
    },
  };
  return updated;
}

/**
 * Serialize a SandboxPolicy to a YAML string suitable for `openshell policy set`.
 */
export function serializePolicy(policy: SandboxPolicy): string {
  return yaml.dump(policy, {
    indent: 2,
    lineWidth: -1,
    noRefs: true,
    quotingType: '"',
    forceQuotes: false,
  });
}
