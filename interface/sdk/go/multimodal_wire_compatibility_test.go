package hms

import (
	"bytes"
	"encoding/json"
	"testing"
)

// Keep the pre-7.10 generated request names source-compatible while exposing
// the additive include_multimodal query builder.
var _ ApiGetVersionRequest = MonitoringAPIGetVersionRequest{}
var _ ApiHealthEndpointHealthGetRequest = MonitoringAPIHealthEndpointHealthGetRequest{}
var _ ApiMetricsEndpointMetricsGetRequest = MonitoringAPIMetricsEndpointMetricsGetRequest{}

type legacyFeaturesInfo struct {
	Observations  bool `json:"observations"`
	Mcp           bool `json:"mcp"`
	Worker        bool `json:"worker"`
	BankConfigApi bool `json:"bank_config_api"`
	FileUploadApi bool `json:"file_upload_api"`
}

type legacyVersionResponse struct {
	ApiVersion string             `json:"api_version"`
	Features   legacyFeaturesInfo `json:"features"`
}

func decodeLegacyVersion(payload []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var response legacyVersionResponse
	return decoder.Decode(&response)
}

func TestLegacyStrictClientAcceptsDefaultNewVersionWire(t *testing.T) {
	legacyWire := []byte(`{"api_version":"0.6.1","features":{"observations":false,"mcp":true,"worker":true,"bank_config_api":false,"file_upload_api":true}}`)
	if err := decodeLegacyVersion(legacyWire); err != nil {
		t.Fatalf("legacy strict client rejected the default new-server wire: %v", err)
	}

	optInWire := []byte(`{"api_version":"0.6.1","features":{"observations":false,"mcp":true,"worker":true,"bank_config_api":false,"file_upload_api":true,"multimodal_image":true,"multimodal_video":false,"multimodal_live_verified":false}}`)
	if err := decodeLegacyVersion(optInWire); err == nil {
		t.Fatal("frozen legacy decoder unexpectedly accepted opt-in capability fields")
	}
}

func TestNewClientDefaultsMissingOldServerCapabilitiesToFalse(t *testing.T) {
	payload := []byte(`{"observations":false,"mcp":true,"worker":true,"bank_config_api":false,"file_upload_api":true}`)
	var features FeaturesInfo
	if err := json.Unmarshal(payload, &features); err != nil {
		t.Fatalf("new client rejected old-server features: %v", err)
	}
	if features.GetMultimodalImage() || features.GetMultimodalVideo() || features.GetMultimodalLiveVerified() {
		t.Fatal("missing old-server multimodal capabilities must read as false")
	}
}

func TestNewClientPreservesLegacyAndTypedOperationMetadata(t *testing.T) {
	payload := []byte(`{
		"operation_id":"00000000-0000-0000-0000-000000000001",
		"status":"completed",
		"result_metadata":{
			"original_filename":"screen.png",
			"multimodal":{
				"asset_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				"media_kind":"image",
				"stage":"recall_ready",
				"recall_ready":true,
				"retryable":false,
				"input_tokens":3000000000
			}
		}
	}`)

	var operation OperationStatusResponse
	if err := json.Unmarshal(payload, &operation); err != nil {
		t.Fatalf("new client rejected typed operation metadata: %v", err)
	}
	metadata, ok := operation.GetResultMetadataOk()
	if !ok || metadata == nil {
		t.Fatal("typed result metadata was not set")
	}
	if metadata.AdditionalProperties["original_filename"] != "screen.png" {
		t.Fatalf("legacy metadata was not preserved: %#v", metadata.AdditionalProperties)
	}
	multimodal, ok := metadata.GetMultimodalOk()
	if !ok || multimodal == nil || !multimodal.GetRecallReady() || multimodal.GetMediaKind() != "image" {
		t.Fatalf("typed multimodal metadata did not round-trip: %#v", multimodal)
	}
	if multimodal.GetInputTokens() != int64(3_000_000_000) {
		t.Fatalf("public int64 counter was truncated: %d", multimodal.GetInputTokens())
	}
}
