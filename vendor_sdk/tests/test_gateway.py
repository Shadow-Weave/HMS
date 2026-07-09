import tempfile
import unittest

from fastapi import HTTPException

from hms_vendor_sdk.gateway import GatewayConfig, GatewayState, _redact


class GatewayTest(unittest.TestCase):
    def test_redacts_secrets(self):
        payload = {
            "authorization": "Bearer hms_live_abcdefghijklmnopqrstuvwxyz",
            "email": "user@example.com",
            "nested": {"api_key": "sk-abcdefghijklmnopqrstuvwxyz"},
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


if __name__ == "__main__":
    unittest.main()
