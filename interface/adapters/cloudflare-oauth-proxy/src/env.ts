/**
 * Environment bindings and OAuth helper types.
 */

/**
 * Minimal shape of the helpers exposed by `@cloudflare/workers-oauth-provider`
 * that this worker relies on. Typed explicitly so we can mock the provider in
 * tests without pulling in the real library.
 */
export interface OAuthReqInfo {
  clientId: string;
  scope?: string | string[];
  [key: string]: unknown;
}

export interface OAuthHelpers {
  parseAuthRequest(request: Request): Promise<OAuthReqInfo>;
  completeAuthorization(opts: {
    request: OAuthReqInfo;
    userId: string;
    metadata?: Record<string, unknown>;
    scope?: string | string[];
    props?: Record<string, unknown>;
  }): Promise<{ redirectTo: string }>;
}

export interface Env {
  OAUTH_KV: KVNamespace;
  OAUTH_PROVIDER: OAuthHelpers;
  HMS_ORIGIN: string;
  ALLOWED_EMAIL: string;
  SESSION_SECRET: string;
  PROXY_SECRET: string;
  HMS_API_TOKEN: string;
}
