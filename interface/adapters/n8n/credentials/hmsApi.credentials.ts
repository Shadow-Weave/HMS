import type {
  IAuthenticateGeneric,
  ICredentialTestRequest,
  ICredentialType,
  INodeProperties,
} from "n8n-workflow";

/**
 * Credentials for the HMS memory API.
 *
 * Defaults to HMS Cloud. For self-hosted instances, change the API URL
 * to your deployment (e.g. http://localhost:8888).
 */
export class HMSApi implements ICredentialType {
  name = "hmsApi";
  displayName = "HMS API";
  documentationUrl = "https://docs.hms.local/developer/api/quickstart";
  properties: INodeProperties[] = [
    {
      displayName: "API URL",
      name: "apiUrl",
      type: "string",
      default: "https://api.hms.local",
      description:
        "Base URL of the HMS API. Defaults to HMS Cloud; change for self-hosted instances.",
      required: true,
    },
    {
      displayName: "API Key",
      name: "apiKey",
      type: "string",
      typeOptions: { password: true },
      default: "",
      description:
        'API key for HMS Cloud (begins with "hsk_"). Leave blank for unauthenticated self-hosted instances.',
    },
  ];

  authenticate: IAuthenticateGeneric = {
    type: "generic",
    properties: {
      headers: {
        Authorization: '={{ $credentials.apiKey ? "Bearer " + $credentials.apiKey : "" }}',
      },
    },
  };

  // Test against /health — works for both Cloud and self-hosted
  test: ICredentialTestRequest = {
    request: {
      baseURL: "={{ $credentials.apiUrl }}",
      url: "/health",
      method: "GET",
    },
  };
}
