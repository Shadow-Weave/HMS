from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .client import HMSVendorClient, HMSVendorError
from .models import RecallBundle, RecallItem
from .organizer import EvidenceOrganizer


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"hms_(?:live|test)_[A-Za-z0-9_\-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{12,}", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
]


class GatewayConfig(BaseModel):
    internal_base_url: str = Field(default="http://127.0.0.1:18080")
    internal_api_key: str | None = None
    external_api_keys: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int = 60
    daily_quota: int = 1000
    audit_log_path: str = ".aaaLOG/vendor_gateway_audit.jsonl"

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        keys = os.getenv("HMS_GATEWAY_API_KEYS") or os.getenv("HMS_GATEWAY_API_KEY") or ""
        external_api_keys = [key.strip() for key in keys.split(",") if key.strip()]
        return cls(
            internal_base_url=os.getenv("HMS_INTERNAL_BASE_URL", "http://127.0.0.1:18080"),
            internal_api_key=os.getenv("HMS_INTERNAL_API_KEY") or os.getenv("HMS_API_TENANT_API_KEY"),
            external_api_keys=external_api_keys,
            rate_limit_per_minute=int(os.getenv("HMS_GATEWAY_RATE_LIMIT_PER_MINUTE", "60")),
            daily_quota=int(os.getenv("HMS_GATEWAY_DAILY_QUOTA", "1000")),
            audit_log_path=os.getenv("HMS_GATEWAY_AUDIT_LOG", ".aaaLOG/vendor_gateway_audit.jsonl"),
        )


class VendorPipelineRequest(BaseModel):
    bank_id: str
    sessions: list[dict[str, Any]]
    question: str
    question_date: str | None = None
    create_bank: bool = False
    reset_bank: bool = False
    bank_profile: dict[str, Any] = Field(default_factory=dict)
    retain_async: bool = False
    wait_for_retain: bool = True
    recall_budget: str = "mid"
    organize: bool = True


class VendorRecallRequest(BaseModel):
    bank_id: str
    question: str
    question_date: str | None = None
    recall_budget: str = "mid"
    max_tokens: int = 4096


class VendorOrganizeRequest(BaseModel):
    question: str
    question_date: str | None = None
    recall_response: dict[str, Any]


class GatewayState:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.client = HMSVendorClient(
            base_url=config.internal_base_url,
            api_key=config.internal_api_key,
            timeout=float(os.getenv("HMS_GATEWAY_INTERNAL_TIMEOUT", "600")),
        )
        self.organizer = EvidenceOrganizer()
        self.lock = threading.Lock()
        self.rate_windows: dict[str, deque[float]] = defaultdict(deque)
        self.daily_usage: dict[tuple[str, str], int] = defaultdict(int)
        self.audit_log_path = Path(config.audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def check_access(self, authorization: str | None) -> str:
        if not self.config.external_api_keys:
            raise HTTPException(status_code=503, detail="Gateway API keys are not configured.")
        token = _extract_bearer(authorization)
        if token not in self.config.external_api_keys:
            raise HTTPException(status_code=401, detail="Invalid gateway API key.")
        key_hash = _hash_key(token)
        self._check_rate_limit(key_hash)
        self._check_quota(key_hash)
        return key_hash

    def _check_rate_limit(self, key_hash: str) -> None:
        now = time.time()
        with self.lock:
            window = self.rate_windows[key_hash]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.config.rate_limit_per_minute:
                raise HTTPException(status_code=429, detail="Rate limit exceeded.")
            window.append(now)

    def _check_quota(self, key_hash: str) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.lock:
            usage_key = (key_hash, today)
            if self.daily_usage[usage_key] >= self.config.daily_quota:
                raise HTTPException(status_code=429, detail="Daily quota exceeded.")
            self.daily_usage[usage_key] += 1

    def audit(self, event: dict[str, Any]) -> None:
        event = _redact(event)
        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    state = GatewayState(config or GatewayConfig.from_env())
    app = FastAPI(title="HMS Vendor Gateway", version="0.1.0")

    def auth_dependency(authorization: str | None = Header(default=None)) -> str:
        return state.check_access(authorization)

    @app.exception_handler(HMSVendorError)
    async def hms_vendor_error_handler(request: Request, exc: HMSVendorError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "gateway": "hms-vendor-gateway",
            "internal_base_url": state.config.internal_base_url,
            "auth_configured": bool(state.config.external_api_keys),
        }

    @app.post("/v1/vendor/pipeline")
    def pipeline(payload: VendorPipelineRequest, key_hash: str = Depends(auth_dependency)):
        started = time.time()
        result = state.client.pipeline(
            bank_id=payload.bank_id,
            sessions=payload.sessions,
            question=payload.question,
            question_date=payload.question_date,
            create_bank=payload.create_bank,
            reset_bank=payload.reset_bank,
            bank_profile=payload.bank_profile,
            retain_async=payload.retain_async,
            wait_for_retain=payload.wait_for_retain,
            recall_budget=payload.recall_budget,
            organize=payload.organize,
        ).to_dict()
        state.audit(
            {
                "action": "pipeline",
                "key_hash": key_hash,
                "bank_id": payload.bank_id,
                "sessions": len(payload.sessions),
                "question": payload.question,
                "duration_ms": int((time.time() - started) * 1000),
                "recall_results": len(result.get("recall_bundle", {}).get("results", [])),
                "ledger_rows": len((result.get("evidence_packet") or {}).get("ledger_rows", [])),
            }
        )
        return result

    @app.post("/v1/vendor/recall")
    def recall(payload: VendorRecallRequest, key_hash: str = Depends(auth_dependency)):
        started = time.time()
        result = state.client.recall(
            bank_id=payload.bank_id,
            question=payload.question,
            question_date=payload.question_date,
            budget=payload.recall_budget,
            max_tokens=payload.max_tokens,
        ).to_dict()
        state.audit(
            {
                "action": "recall",
                "key_hash": key_hash,
                "bank_id": payload.bank_id,
                "question": payload.question,
                "duration_ms": int((time.time() - started) * 1000),
                "recall_results": len(result.get("results", [])),
            }
        )
        return result

    @app.post("/v1/vendor/organize")
    def organize(payload: VendorOrganizeRequest, key_hash: str = Depends(auth_dependency)):
        rows = [RecallItem.from_dict(row) for row in payload.recall_response.get("results", [])]
        recall_bundle = RecallBundle(
            bank_id=payload.recall_response.get("bank_id", "external"),
            question=payload.question,
            question_date=payload.question_date,
            results=rows,
            trace=payload.recall_response.get("trace"),
            entities=payload.recall_response.get("entities"),
            chunks=payload.recall_response.get("chunks"),
            source_facts=payload.recall_response.get("source_facts"),
            raw_response=payload.recall_response,
        )
        packet = state.organizer.organize(
            payload.question,
            recall_bundle,
            question_date=payload.question_date,
        ).to_dict()
        state.audit(
            {
                "action": "organize",
                "key_hash": key_hash,
                "question": payload.question,
                "recall_results": len(rows),
                "ledger_rows": len(packet.get("ledger_rows", [])),
            }
        )
        return packet

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HMS vendor gateway.")
    parser.add_argument("--host", default=os.getenv("HMS_GATEWAY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("HMS_GATEWAY_PORT", "18081")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required.")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def _hash_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("<redacted>" if _looks_secret_key(key) else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_PATTERNS:
            redacted = pattern.sub("<redacted>", redacted)
        if len(redacted) > 600:
            redacted = redacted[:600].rstrip() + "..."
        return redacted
    return value


def _looks_secret_key(key: str) -> bool:
    key_lower = key.lower()
    return any(marker in key_lower for marker in ("api_key", "authorization", "token", "secret", "password"))


if __name__ == "__main__":
    main()
