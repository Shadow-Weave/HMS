import tempfile
import unittest

from fastapi import HTTPException

from hms_vendor_sdk.gateway import GatewayConfig, GatewayState, _redact


class GatewayTest(unittest.TestCase):
    def _state(self, **overrides):
        config = GatewayConfig(
            external_api_keys=["hms_live_test_key"],
            audit_log_path=overrides.pop("audit_log_path", "/tmp/hms_vendor_gateway_test_audit.jsonl"),
            check_internal_health=False,
            **overrides,
        )
        return GatewayState(config)

    def test_redacts_secrets(self):
        payload = {
            "authorization": "Bearer hms_live_abcdefghijklmnopqrstuvwxyz",
            "email": "user@example.com",
            "nested": {"api_key": "provider-key-placeholder-value"},
        }

        redacted = _redact(payload)

        self.assertEqual(redacted["authorization"], "<redacted>")
        self.assertEqual(redacted["email"], "<redacted>")
        self.assertEqual(redacted["nested"]["api_key"], "<redacted>")

    def test_rate_limit_blocks_after_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = GatewayState(
                GatewayConfig(
                    external_api_keys=["hms_live_test_key"],
                    rate_limit_per_minute=1,
                    daily_quota=10,
                    audit_log_path=f"{tmpdir}/audit.jsonl",
                )
            )

            state.check_access("Bearer hms_live_test_key")
            with self.assertRaises(HTTPException) as ctx:
                state.check_access("Bearer hms_live_test_key")

            self.assertEqual(ctx.exception.status_code, 429)

    def test_quota_blocks_after_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = GatewayState(
                GatewayConfig(
                    external_api_keys=["hms_live_test_key"],
                    rate_limit_per_minute=10,
                    daily_quota=1,
                    audit_log_path=f"{tmpdir}/audit.jsonl",
                )
            )

            state.check_access("Bearer hms_live_test_key")
            with self.assertRaises(HTTPException) as ctx:
                state.check_access("Bearer hms_live_test_key")

            self.assertEqual(ctx.exception.status_code, 429)

    def test_scopes_bank_id_by_gateway_key(self):
        state = self._state()

        key_hash = state.check_access("Bearer hms_live_test_key")
        internal_bank_id = state.internal_bank_id("vendor-demo", key_hash)

        self.assertTrue(internal_bank_id.startswith(f"vendor_{key_hash}_"))
        self.assertNotEqual(internal_bank_id, "vendor-demo")

    def test_can_disable_bank_id_scoping_for_private_deployments(self):
        state = self._state(scope_bank_ids=False)

        key_hash = state.check_access("Bearer hms_live_test_key")
        internal_bank_id = state.internal_bank_id("vendor-demo", key_hash)

        self.assertEqual(internal_bank_id, "vendor-demo")

    def test_rejects_unsafe_bank_id(self):
        state = self._state()

        with self.assertRaises(HTTPException) as ctx:
            state.internal_bank_id("../vendor-demo", "abc123")

        self.assertEqual(ctx.exception.status_code, 400)

    def test_public_result_rewrites_internal_bank_id(self):
        state = self._state()

        result = {
            "bank_id": "vendor_abcd1234_demo",
            "nested": {"bank_id": "vendor_abcd1234_demo"},
        }
        public = state.public_result(
            result,
            public_bank_id="demo",
            internal_bank_id="vendor_abcd1234_demo",
        )

        self.assertEqual(public["bank_id"], "demo")
        self.assertEqual(public["nested"]["bank_id"], "demo")


if __name__ == "__main__":
    unittest.main()
