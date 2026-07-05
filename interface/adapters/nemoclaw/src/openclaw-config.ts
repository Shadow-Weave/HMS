import { readFile, writeFile, rename } from "fs/promises";
import { join, dirname } from "path";
import { homedir } from "os";
import { randomBytes } from "crypto";

const CONFIG_PATH = join(homedir(), ".openclaw", "openclaw.json");

export interface HMSPluginConfig {
  hmsApiUrl: string;
  hmsApiToken: string;
  llmProvider: string;
  dynamicBankId: boolean;
  bankIdPrefix: string;
}

export interface OpenClawConfig {
  plugins?: {
    slots?: Record<string, string>;
    entries?: Record<string, { enabled: boolean; config?: Record<string, unknown> }>;
    installs?: Record<string, unknown>;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export async function readOpenClawConfig(configPath = CONFIG_PATH): Promise<OpenClawConfig> {
  let raw: string;
  try {
    raw = await readFile(configPath, "utf8");
  } catch (err: unknown) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      throw new Error(
        `OpenClaw config not found at ${configPath}.\n` +
          `Run \`openclaw\` once to initialize it, then re-run setup.`
      );
    }
    throw err;
  }
  return JSON.parse(raw) as OpenClawConfig;
}

export function mergePluginConfig(
  config: OpenClawConfig,
  pluginConfig: HMSPluginConfig
): OpenClawConfig {
  const plugins = config.plugins ?? {};
  const entries = plugins.entries ?? {};
  const existing = entries["hms-openclaw"] ?? { enabled: true };

  return {
    ...config,
    plugins: {
      ...plugins,
      slots: {
        ...(plugins.slots ?? {}),
        memory: "hms-openclaw",
      },
      entries: {
        ...entries,
        "hms-openclaw": {
          ...existing,
          enabled: true,
          config: {
            ...(existing.config ?? {}),
            hmsApiUrl: pluginConfig.hmsApiUrl,
            hmsApiToken: pluginConfig.hmsApiToken,
            llmProvider: pluginConfig.llmProvider,
            dynamicBankId: pluginConfig.dynamicBankId,
            bankIdPrefix: pluginConfig.bankIdPrefix,
          },
        },
      },
      installs: {
        ...(plugins.installs ?? {}),
        "hms-openclaw": {
          source: "npm",
          version: "latest",
          installedAt: new Date().toISOString(),
        },
      },
    },
  };
}

export async function writeOpenClawConfig(
  config: OpenClawConfig,
  configPath = CONFIG_PATH
): Promise<void> {
  const contents = JSON.stringify(config, null, 2) + "\n";
  const tmp = `${configPath}.${randomBytes(6).toString("hex")}.tmp`;
  await writeFile(tmp, contents, "utf8");
  await rename(tmp, configPath);
}

export async function applyPluginConfig(
  pluginConfig: HMSPluginConfig,
  configPath = CONFIG_PATH
): Promise<void> {
  const current = await readOpenClawConfig(configPath);
  const updated = mergePluginConfig(current, pluginConfig);
  await writeOpenClawConfig(updated, configPath);
}

export { CONFIG_PATH };
export { dirname };
