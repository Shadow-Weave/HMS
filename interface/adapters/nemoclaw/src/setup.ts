import { execFile } from "child_process";
import { promisify } from "util";
import { writeFile, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import { randomBytes } from "crypto";
import type { CliArgs } from "./types.js";
import { readSandboxPolicy } from "./policy-reader.js";
import { hasHMSPolicy, mergeHMSPolicy, serializePolicy } from "./policy-writer.js";
import { applyPluginConfig } from "./openclaw-config.js";

const execFileAsync = promisify(execFile);

function log(msg: string) {
  process.stdout.write(`${msg}\n`);
}

function step(n: number, msg: string) {
  log(`\n[${n}] ${msg}`);
}

async function which(bin: string): Promise<boolean> {
  try {
    await execFileAsync("which", [bin]);
    return true;
  } catch {
    return false;
  }
}

export async function runSetup(args: CliArgs): Promise<void> {
  log("\nhms-nemoclaw setup");
  log("─".repeat(40));

  // Step 0 — Preflight
  step(0, "Preflight checks...");
  const [hasOpenshell, hasOpenclaw] = await Promise.all([which("openshell"), which("openclaw")]);
  if (!hasOpenshell) {
    throw new Error("`openshell` not found on PATH. Install it from https://openshell.ai");
  }
  if (!hasOpenclaw) {
    throw new Error("`openclaw` not found on PATH. Install it from https://openclaw.ai");
  }
  log("  ✓ openshell found");
  log("  ✓ openclaw found");

  // Step 1 — Install hms-openclaw plugin
  if (!args.skipPluginInstall) {
    step(1, "Installing @hms-memory/hms-openclaw plugin...");
    if (args.dryRun) {
      log("  [dry-run] would run: openclaw plugins install @hms-memory/hms-openclaw");
    } else {
      const { stdout } = await execFileAsync("openclaw", [
        "plugins",
        "install",
        "@hms-memory/hms-openclaw",
      ]);
      log(stdout.trim() || "  ✓ Plugin installed");
    }
  } else {
    step(1, "Skipping plugin install (--skip-plugin-install)");
  }

  // Step 2 — Configure ~/.openclaw/openclaw.json
  step(2, "Configuring plugin in ~/.openclaw/openclaw.json...");
  const pluginConfig = {
    hmsApiUrl: args.apiUrl,
    hmsApiToken: args.apiToken,
    llmProvider: "claude-code",
    dynamicBankId: false,
    bankIdPrefix: args.bankPrefix,
  };
  if (args.dryRun) {
    log(`  [dry-run] would write plugin config to ~/.openclaw/openclaw.json`);
    log(`  config: ${JSON.stringify(pluginConfig, null, 4).split("\n").join("\n  ")}`);
  } else {
    await applyPluginConfig(pluginConfig);
    log(`  ✓ Plugin config written (bank: ${args.bankPrefix}-openclaw)`);
  }

  // Step 3 — Apply OpenShell network policy
  if (!args.skipPolicy) {
    step(3, `Applying HMS network policy to sandbox "${args.sandbox}"...`);

    const currentPolicy = await readSandboxPolicy(args.sandbox);

    if (hasHMSPolicy(currentPolicy)) {
      log("  ✓ HMS policy already present — skipping");
    } else {
      const updatedPolicy = mergeHMSPolicy(currentPolicy);
      const policyYaml = serializePolicy(updatedPolicy);

      if (args.dryRun) {
        log("  [dry-run] would apply policy:");
        log(
          policyYaml
            .split("\n")
            .map((l) => `    ${l}`)
            .join("\n")
        );
      } else {
        const tmpFile = join(tmpdir(), `hms-policy-${randomBytes(6).toString("hex")}.yaml`);
        try {
          await writeFile(tmpFile, policyYaml, "utf8");
          const { stdout } = await execFileAsync("openshell", [
            "policy",
            "set",
            args.sandbox,
            "--policy",
            tmpFile,
            "--wait",
          ]);
          log(stdout.trim() || `  ✓ Policy applied to sandbox "${args.sandbox}"`);
        } finally {
          await rm(tmpFile, { force: true });
        }
      }
    }
  } else {
    step(3, "Skipping policy update (--skip-policy)");
    log("  Add the following block to your sandbox network_policies manually:");
    log("");
    log("    hms:");
    log("      name: hms");
    log("      endpoints:");
    log("        - host: api.hms.local");
    log("          port: 443");
    log("          protocol: rest");
    log("          tls: terminate");
    log("          enforcement: enforce");
    log("          rules:");
    log("            - allow: { method: GET, path: /** }");
    log("            - allow: { method: POST, path: /** }");
    log("            - allow: { method: PUT, path: /** }");
    log("      binaries:");
    log("        - path: /usr/local/bin/openclaw");
  }

  // Step 4 — Restart gateway
  step(4, "Restarting OpenClaw gateway...");
  if (args.dryRun) {
    log("  [dry-run] would run: openclaw gateway restart");
  } else {
    await execFileAsync("openclaw", ["gateway", "restart"]);
    log("  ✓ Gateway restarted");
  }

  log("\n" + "─".repeat(40));
  log("✓ Setup complete!\n");
  log(`  Bank ID: ${args.bankPrefix}-openclaw`);
  log(`  API URL: ${args.apiUrl}`);
  log("");
  log("  Watch gateway logs to confirm:");
  log("    grep HMS ~/.openclaw/logs/gateway.log | tail -5");
  log("  Expected: [HMS] ✓ Ready (external API mode)");
  log("");
  log("  Test memory retention:");
  log(`    openclaw agent --agent main --session-id test-1 -m "My name is Ben."`);
  log(`    openclaw agent --agent main --session-id test-2 -m "What do you remember about me?"`);
}
