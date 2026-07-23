import type {
  FeaturesInfo,
  GetVersionData,
  OperationResultMetadata,
  OperationStatusResponse,
  VersionResponse,
} from "../generated/types.gen";

type LegacyFeaturesInfo = Pick<
  FeaturesInfo,
  "observations" | "mcp" | "worker" | "bank_config_api" | "file_upload_api"
>;

type LegacyVersionResponse = {
  api_version: string;
  features: LegacyFeaturesInfo;
};

const oldServerFeatures: FeaturesInfo = {
  observations: false,
  mcp: true,
  worker: true,
  bank_config_api: false,
  file_upload_api: true,
};

const newServerVersion = {
  api_version: "0.6.1",
  features: {
    ...oldServerFeatures,
    multimodal_image: true,
    multimodal_video: false,
    multimodal_live_verified: false,
  },
} satisfies VersionResponse;

// Structural decoding used by the old TypeScript SDK ignores additive fields.
const legacyClientView: LegacyVersionResponse = newServerVersion;

const metadata = {
  legacy_debug_key: 7,
  multimodal: {
    asset_sha256: "a".repeat(64),
    media_kind: "image",
    stage: "recall_ready",
    recall_ready: true,
    retryable: false,
  },
} satisfies OperationResultMetadata;

const operation = {
  operation_id: "00000000-0000-0000-0000-000000000001",
  status: "completed",
  result_metadata: metadata,
} satisfies OperationStatusResponse;

const versionQuery = {
  query: { include_multimodal: true },
  url: "/version",
} satisfies GetVersionData;

describe("multimodal wire compatibility", () => {
  test("keeps old-server defaults and additive client shapes", () => {
    expect(oldServerFeatures.multimodal_image).toBeUndefined();
    expect(legacyClientView.features.file_upload_api).toBe(true);
    expect(operation.result_metadata?.multimodal?.recall_ready).toBe(true);
    expect(versionQuery.query.include_multimodal).toBe(true);
  });
});
