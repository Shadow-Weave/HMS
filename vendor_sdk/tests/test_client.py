import json
import unittest
from unittest.mock import patch

from hms_vendor_sdk import HMSVendorClient, HMSVendorError, RecallBundle, RecallItem, SessionRecord


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class HMSVendorClientTest(unittest.TestCase):
    def setUp(self):
        self.client = HMSVendorClient(base_url="http://localhost:8888", api_key="token")

    def test_retain_sessions_flattens_messages(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["method"] = req.get_method()
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResponse({"success": True, "items_count": 1, "async": False})

        session = SessionRecord(
            session_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            messages=[
                {"role": "user", "content": "Need to pick up laundry."},
                {"role": "assistant", "content": "Recorded."},
            ],
            metadata={"tenant": "demo"},
            tags=["user_a"],
        )

        with patch("hms_vendor_sdk.client.request.urlopen", side_effect=fake_urlopen):
            summary = self.client.retain_sessions("bank-1", [session])

        self.assertTrue(summary.success)
        self.assertEqual(captured["method"], "POST")
        self.assertTrue(captured["url"].endswith("/v1/default/banks/bank-1/memories"))
        self.assertEqual(captured["body"]["items"][0]["document_id"], "s1")
        self.assertIn("[USER]\nNeed to pick up laundry.", captured["body"]["items"][0]["content"])
        self.assertIn("[ASSISTANT]\nRecorded.", captured["body"]["items"][0]["content"])
        self.assertEqual(captured["body"]["items"][0]["metadata"]["session_id"], "s1")

    def test_bank_id_path_segment_is_encoded(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            return FakeResponse({"success": True, "items_count": 1, "async": False})

        with patch("hms_vendor_sdk.client.request.urlopen", side_effect=fake_urlopen):
            self.client.retain_sessions(
                "vendor/demo",
                [{"session_id": "s1", "messages": [{"role": "user", "content": "A"}]}],
            )

        self.assertTrue(captured["url"].endswith("/v1/default/banks/vendor%2Fdemo/memories"))

    def test_recall_normalizes_results(self):
        def fake_urlopen(req, timeout):
            payload = {
                "results": [
                    {
                        "id": "m1",
                        "text": "Need to pick up laundry.",
                        "type": "experience",
                        "document_id": "s1",
                        "score": 0.98,
                    }
                ],
                "trace": {"num_results": 1},
            }
            return FakeResponse(payload)

        with patch("hms_vendor_sdk.client.request.urlopen", side_effect=fake_urlopen):
            recall = self.client.recall("bank-1", "What do I need to pick up?")

        self.assertEqual(recall.bank_id, "bank-1")
        self.assertEqual(len(recall.results), 1)
        self.assertEqual(recall.results[0].id, "m1")
        self.assertEqual(recall.results[0].extra["score"], 0.98)
        self.assertEqual(recall.trace["num_results"], 1)

    def test_pipeline_runs_create_retain_recall(self):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append((req.get_method(), req.full_url))
            if req.get_method() == "PUT":
                return FakeResponse({"bank_id": "bank-1"})
            if req.full_url.endswith("/memories"):
                return FakeResponse({"success": True, "items_count": 1, "async": False})
            return FakeResponse({"results": [], "trace": {"num_results": 0}})

        with patch("hms_vendor_sdk.client.request.urlopen", side_effect=fake_urlopen):
            result = self.client.pipeline(
                bank_id="bank-1",
                sessions=[
                    {
                        "session_id": "s1",
                        "messages": [{"role": "user", "content": "Need to pick up laundry."}],
                    }
                ],
                question="What do I need to pick up?",
                create_bank=True,
            )

        self.assertEqual(result.bank_id, "bank-1")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], "PUT")
        self.assertEqual(calls[1][0], "POST")
        self.assertEqual(calls[2][0], "POST")
        self.assertIsNotNone(result.evidence_packet)

    def test_pipeline_waits_for_async_retain_before_recall(self):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append((req.get_method(), req.full_url))
            if req.get_method() == "PUT":
                return FakeResponse({"bank_id": "bank-1"})
            if req.full_url.endswith("/memories"):
                return FakeResponse(
                    {
                        "success": True,
                        "items_count": 1,
                        "async": True,
                        "operation_id": "00000000-0000-0000-0000-000000000001",
                    }
                )
            if "/operations/" in req.full_url:
                return FakeResponse(
                    {
                        "operation_id": "00000000-0000-0000-0000-000000000001",
                        "status": "completed",
                        "operation_type": "retain",
                    }
                )
            return FakeResponse({"results": [], "trace": {"num_results": 0}})

        with patch("hms_vendor_sdk.client.request.urlopen", side_effect=fake_urlopen):
            self.client.pipeline(
                bank_id="bank-1",
                sessions=[
                    {
                        "session_id": "s1",
                        "messages": [{"role": "user", "content": "Need to pick up laundry."}],
                    }
                ],
                question="What do I need to pick up?",
                create_bank=True,
                retain_async=True,
                poll_interval=0,
            )

        operation_call_index = next(i for i, call in enumerate(calls) if "/operations/" in call[1])
        recall_call_index = next(i for i, call in enumerate(calls) if call[1].endswith("/memories/recall"))
        self.assertLess(operation_call_index, recall_call_index)

    def test_retain_sessions_rejects_duplicate_session_ids(self):
        sessions = [
            {"session_id": "s1", "messages": [{"role": "user", "content": "A"}]},
            {"session_id": "s1", "messages": [{"role": "user", "content": "B"}]},
        ]

        with self.assertRaises(HMSVendorError):
            self.client.retain_sessions("bank-1", sessions)

    def test_retain_stringifies_metadata_values(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResponse({"success": True, "items_count": 1, "async": False})

        with patch("hms_vendor_sdk.client.request.urlopen", side_effect=fake_urlopen):
            self.client.retain_memory(
                bank_id="bank-1",
                content="User likes tea.",
                metadata={"rank": 1, "nested": {"source": "demo"}},
            )

        metadata = captured["body"]["items"][0]["metadata"]
        self.assertEqual(metadata["rank"], "1")
        self.assertEqual(metadata["nested"], '{"source": "demo"}')

    def test_organize_builds_evidence_packet(self):
        recall = RecallBundle(
            bank_id="bank-1",
            question="How many items do I need to pick up?",
            question_date="2024-03-10T00:00:00Z",
            results=[
                RecallItem(
                    id="m1",
                    text="The user needs to pick up a navy blue blazer from dry cleaning.",
                    type="experience",
                    document_id="s1",
                    mentioned_at="2024-03-01T10:00:00Z",
                )
            ],
        )
        packet = self.client.organize(
            "How many items do I need to pick up?",
            recall,
            question_date="2024-03-10T00:00:00Z",
        )

        self.assertGreaterEqual(len(packet.controls), 1)
        self.assertIn("Structured Evidence Ledger", packet.answer_ready_context)


if __name__ == "__main__":
    unittest.main()
