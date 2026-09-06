"""HMS REST API client.

Communicates with a HMS server via HTTP. Mirrors the HTTP mode of the
Openclaw HMSClient (client.js), adapted for Python stdlib.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_TIMEOUT = 15  # seconds
HEALTH_CHECK_RETRIES = 3
HEALTH_CHECK_DELAY = 2  # seconds


def _plugin_version() -> str:
    """Read the plugin version from settings.json (single source of truth)."""
    manifest = Path(__file__).resolve().parents[2] / "settings.json"
    try:
        return json.loads(manifest.read_text()).get("version", "0.0.0")
    except (OSError, ValueError):
        return "0.0.0"


# Sent on every request so self-hosted deployments behind Cloudflare (or any
# reverse proxy with UA-based bot filtering) don't block the stdlib default
# "Python-urllib/X.Y", which trips Cloudflare error 1010.
USER_AGENT = f"hms-codex/{_plugin_version()}"


def _validate_api_url(url: str) -> str:
    """Validate and normalize the API URL. Reject non-HTTP schemes."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"HMS API URL must use http or https, got: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError(f"HMS API URL has no hostname: {url!r}")
    return url.rstrip("/")


class HMSClient:
    """HTTP client for the HMS API."""

    def __init__(self, api_url: str, api_token: Optional[str] = None):
        self.api_url = _validate_api_url(api_url)
        self.api_token = api_token

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _request(self, method: str, path: str, body: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
        url = f"{self.api_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode()
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code} from {url}: {body_text}") from e

    def health_check(self, timeout: int = 5) -> bool:
        """Check if the HMS server is reachable.

        Mirrors Openclaw's checkExternalApiHealth: retries up to 3 times
        with 2s delay between attempts.
        """
        import time

        for attempt in range(1, HEALTH_CHECK_RETRIES + 1):
            try:
                url = f"{self.api_url}/health"
                req = urllib.request.Request(url, headers=self._headers(), method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            if attempt < HEALTH_CHECK_RETRIES:
                time.sleep(HEALTH_CHECK_DELAY)
        return False

    def recall(
        self,
        bank_id: str,
        query: str,
        max_tokens: int = 1024,
        budget: str = "mid",
        types: Optional[list] = None,
        timeout: int = 10,
    ) -> dict:
        """Recall memories from a bank.

        Returns the raw API response dict with 'results' list.
        """
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/memories/recall"
        body = {
            "query": query,
            "max_tokens": max_tokens,
        }
        if budget:
            body["budget"] = budget
        if types:
            body["types"] = types
        return self._request("POST", path, body, timeout=timeout)

    def retain(
        self,
        bank_id: str,
        content: str,
        document_id: str = "conversation",
        context: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
        timeout: int = 15,
    ) -> dict:
        """Retain content into a bank's memory.

        Posts with async=true so the server processes in the background.
        The context field helps HMS cluster memories by provenance
        (e.g. "claude-code" vs manual retains).
        """
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/memories"
        item = {
            "content": content,
            "document_id": document_id,
            "metadata": metadata or {},
        }
        if context:
            item["context"] = context
        if tags:
            item["tags"] = tags
        body = {
            "items": [item],
            "async": True,
        }
        return self._request("POST", path, body, timeout=timeout)

    def set_bank_mission(
        self, bank_id: str, mission: str, retain_mission: Optional[str] = None, timeout: int = 15
    ) -> dict:
        """Set the mission/persona for a bank.

        Uses PATCH /banks/{id}/config with reflect_mission and retain_mission.
        The old PUT /banks/{id} with 'mission' field is deprecated in v0.4.19.
        """
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/config"
        updates = {"reflect_mission": mission}
        if retain_mission:
            updates["retain_mission"] = retain_mission
        return self._request("PATCH", path, {"updates": updates}, timeout=timeout)

    def create_rsi_snapshot(self, bank_id: str, snapshot_id: str, *, parent_snapshot_id=None, manifest=None) -> dict:
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/snapshots"
        return self._request(
            "POST",
            path,
            {
                "snapshot_id": snapshot_id,
                "parent_snapshot_id": parent_snapshot_id,
                "manifest": manifest or {},
            },
            timeout=15,
        )

    def get_rsi_snapshot(self, bank_id: str, snapshot_id: str) -> dict:
        path = (
            f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/snapshots/"
            f"{urllib.parse.quote(snapshot_id, safe='')}"
        )
        return self._request("GET", path, timeout=15)

    def freeze_rsi_snapshot(self, bank_id: str, snapshot_id: str) -> dict:
        path = (
            f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/snapshots/"
            f"{urllib.parse.quote(snapshot_id, safe='')}/freeze"
        )
        return self._request("POST", path, timeout=15)

    def search_rsi_snapshot(self, bank_id: str, snapshot_id: str, namespace: str, query: str, limit: int = 20) -> dict:
        path = (
            f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/snapshots/"
            f"{urllib.parse.quote(snapshot_id, safe='')}/search"
        )
        return self._request(
            "POST",
            path,
            {"namespace": namespace, "query": query, "limit": max(1, min(int(limit), 100))},
            timeout=15,
        )

    def fetch_rsi_snapshot_memory(self, bank_id: str, snapshot_id: str, memory_id: str) -> dict:
        path = (
            f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/snapshots/"
            f"{urllib.parse.quote(snapshot_id, safe='')}/memories/{urllib.parse.quote(memory_id, safe='')}"
        )
        return self._request("GET", path, timeout=15)

    def tag_rsi_memory_namespace(
        self, bank_id: str, memory_ids: list[str], namespace: str, only_unclassified: bool = True
    ) -> dict:
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/memories/namespace"
        return self._request(
            "POST",
            path,
            {"memory_ids": memory_ids, "namespace": namespace, "only_unclassified": only_unclassified},
            timeout=15,
        )

    def stage_rsi_candidate(self, bank_id: str, candidate: dict) -> dict:
        path = f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/candidates"
        return self._request("POST", path, candidate, timeout=15)

    def promote_rsi_candidates(
        self,
        bank_id: str,
        snapshot_id: str,
        candidate_ids: list[str],
        run_id: str,
        candidate_digest: str,
        receipt_token: str,
    ) -> dict:
        path = (
            f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/snapshots/"
            f"{urllib.parse.quote(snapshot_id, safe='')}/promote"
        )
        return self._request(
            "POST",
            path,
            {
                "candidate_ids": candidate_ids,
                "run_id": run_id,
                "candidate_digest": candidate_digest,
                "verifier_receipt_token": receipt_token,
            },
            timeout=15,
        )

    def invalidate_rsi_candidate(self, bank_id: str, candidate_id: str, reason: str) -> dict:
        path = (
            f"/v1/default/banks/{urllib.parse.quote(bank_id, safe='')}/rsi/candidates/"
            f"{urllib.parse.quote(candidate_id, safe='')}/invalidate"
        )
        return self._request("POST", path, {"reason": reason}, timeout=15)
