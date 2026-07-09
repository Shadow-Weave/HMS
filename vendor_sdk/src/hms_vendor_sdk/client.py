from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib import error, request

from .models import (
    EvidencePacket,
    OperationStatus,
    PipelineResult,
    RecallBundle,
    RecallItem,
    RetainSummary,
    SessionMessage,
    SessionRecord,
)
from .organizer import EvidenceOrganizer


class HMSVendorError(RuntimeError):
    pass


class HMSVendorClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.organizer = EvidenceOrganizer()

    @classmethod
    def from_env(cls) -> "HMSVendorClient":
        base_url = os.getenv("HMS_BASE_URL") or os.getenv("HMS_VENDOR_BASE_URL")
        if not base_url:
            raise HMSVendorError("Set HMS_BASE_URL before using HMSVendorClient.from_env().")
        return cls(
            base_url=base_url,
            api_key=os.getenv("HMS_API_KEY") or os.getenv("HMS_VENDOR_API_KEY"),
            timeout=float(os.getenv("HMS_TIMEOUT", "300")),
        )

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def version(self) -> dict[str, Any]:
        return self._request_json("GET", "/version")

    def create_bank(
        self,
        bank_id: str,
        *,
        reflect_mission: str | None = None,
        retain_mission: str | None = None,
        retain_extraction_mode: str | None = None,
        retain_custom_instructions: str | None = None,
        enable_observations: bool | None = None,
        observations_mission: str | None = None,
        background: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "reflect_mission": reflect_mission,
            "retain_mission": retain_mission,
            "retain_extraction_mode": retain_extraction_mode,
            "retain_custom_instructions": retain_custom_instructions,
            "enable_observations": enable_observations,
            "observations_mission": observations_mission,
            "background": background,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        return self._request_json("PUT", f"/v1/default/banks/{bank_id}", payload)

    def delete_bank(self, bank_id: str, *, missing_ok: bool = False) -> dict[str, Any]:
        try:
            return self._request_json("DELETE", f"/v1/default/banks/{bank_id}")
        except HMSVendorError as exc:
            if missing_ok and "failed: 404" in str(exc):
                return {"success": True, "message": f"Bank '{bank_id}' did not exist."}
            raise

    def retain_memory(
        self,
        bank_id: str,
        content: str,
        *,
        timestamp: str | None = None,
        context: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        update_mode: str | None = None,
        retain_async: bool = False,
    ) -> RetainSummary:
        return self.retain_items(
            bank_id=bank_id,
            items=[
                {
                    "content": content,
                    "timestamp": timestamp,
                    "context": context,
                    "document_id": document_id,
                    "metadata": metadata or {},
                    "tags": tags or [],
                    "update_mode": update_mode,
                }
            ],
            retain_async=retain_async,
        )

    def retain_items(
        self,
        bank_id: str,
        items: list[dict[str, Any]],
        *,
        document_tags: list[str] | None = None,
        retain_async: bool = False,
    ) -> RetainSummary:
        if not items:
            raise HMSVendorError("retain_items() requires at least one item.")

        cleaned_items = []
        for item in items:
            cleaned = {key: value for key, value in item.items() if value is not None}
            if "metadata" in cleaned:
                cleaned["metadata"] = self._stringify_metadata(cleaned["metadata"])
            cleaned_items.append(cleaned)

        payload = {
            "items": cleaned_items,
            "async": retain_async,
        }
        if document_tags:
            payload["document_tags"] = document_tags

        response = self._request_json("POST", f"/v1/default/banks/{bank_id}/memories", payload)
        return RetainSummary(
            bank_id=bank_id,
            items_count=int(response.get("items_count", len(cleaned_items))),
            success=bool(response.get("success", False)),
            async_mode=bool(response.get("async", retain_async)),
            operation_id=response.get("operation_id"),
            operation_ids=response.get("operation_ids"),
            usage=response.get("usage"),
            raw_response=response,
        )

    def retain_sessions(
        self,
        bank_id: str,
        sessions: list[SessionRecord | dict[str, Any]],
        *,
        document_tags: list[str] | None = None,
        retain_async: bool = False,
    ) -> RetainSummary:
        self._validate_sessions(sessions)
        items = [self._session_to_item(session) for session in sessions]
        return self.retain_items(
            bank_id=bank_id,
            items=items,
            document_tags=document_tags,
            retain_async=retain_async,
        )

    def recall(
        self,
        bank_id: str,
        question: str,
        *,
        question_date: str | None = None,
        budget: str = "mid",
        max_tokens: int = 4096,
        trace: bool = True,
        include_entities: bool = True,
        max_entity_tokens: int = 500,
        include_chunks: bool = True,
        max_chunk_tokens: int = 8192,
        include_source_facts: bool = False,
        max_source_facts_tokens: int = 4096,
        types: list[str] | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
    ) -> RecallBundle:
        include: dict[str, Any] = {}
        if include_entities:
            include["entities"] = {"max_tokens": max_entity_tokens}
        if include_chunks:
            include["chunks"] = {"max_tokens": max_chunk_tokens}
        if include_source_facts:
            include["source_facts"] = {"max_tokens": max_source_facts_tokens}

        payload = {
            "query": question,
            "types": types,
            "budget": budget,
            "max_tokens": max_tokens,
            "trace": trace,
            "query_timestamp": question_date,
            "include": include,
            "tags": tags,
            "tags_match": tags_match,
        }
        payload = {key: value for key, value in payload.items() if value is not None}

        response = self._request_json("POST", f"/v1/default/banks/{bank_id}/memories/recall", payload)
        results = [RecallItem.from_dict(row) for row in response.get("results", [])]
        return RecallBundle(
            bank_id=bank_id,
            question=question,
            question_date=question_date,
            results=results,
            trace=response.get("trace"),
            entities=response.get("entities"),
            chunks=response.get("chunks"),
            source_facts=response.get("source_facts"),
            raw_response=response,
        )

    def pipeline(
        self,
        bank_id: str,
        sessions: list[SessionRecord | dict[str, Any]],
        question: str,
        *,
        question_date: str | None = None,
        create_bank: bool = False,
        reset_bank: bool = False,
        bank_profile: dict[str, Any] | None = None,
        document_tags: list[str] | None = None,
        retain_async: bool = False,
        wait_for_retain: bool = True,
        wait_timeout: float = 600.0,
        poll_interval: float = 2.0,
        recall_budget: str = "mid",
        organize: bool = True,
    ) -> PipelineResult:
        if reset_bank:
            self.delete_bank(bank_id, missing_ok=True)
        if create_bank:
            self.create_bank(bank_id, **(bank_profile or {}))

        retain_summary = self.retain_sessions(
            bank_id=bank_id,
            sessions=sessions,
            document_tags=document_tags,
            retain_async=retain_async,
        )
        if retain_summary.async_mode and wait_for_retain:
            self.wait_for_retain(
                bank_id=bank_id,
                retain_summary=retain_summary,
                timeout=wait_timeout,
                poll_interval=poll_interval,
            )
        recall_bundle = self.recall(
            bank_id=bank_id,
            question=question,
            question_date=question_date,
            budget=recall_budget,
        )
        evidence_packet = self.organize(question, recall_bundle, question_date=question_date) if organize else None
        return PipelineResult(
            bank_id=bank_id,
            retain_summary=retain_summary,
            recall_bundle=recall_bundle,
            evidence_packet=evidence_packet,
        )

    def organize(
        self,
        question: str,
        recall_bundle: RecallBundle,
        *,
        question_date: str | None = None,
    ) -> EvidencePacket:
        return self.organizer.organize(question, recall_bundle, question_date=question_date)

    def get_operation(self, bank_id: str, operation_id: str, *, include_payload: bool = False) -> OperationStatus:
        query = "?include_payload=true" if include_payload else ""
        response = self._request_json("GET", f"/v1/default/banks/{bank_id}/operations/{operation_id}{query}")
        return OperationStatus(
            operation_id=response["operation_id"],
            status=response["status"],
            operation_type=response.get("operation_type"),
            error_message=response.get("error_message"),
            result_metadata=response.get("result_metadata"),
            raw_response=response,
        )

    def wait_for_operation(
        self,
        bank_id: str,
        operation_id: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ) -> OperationStatus:
        deadline = time.monotonic() + timeout
        while True:
            status = self.get_operation(bank_id, operation_id)
            if status.status == "completed":
                return status
            if status.status in {"failed", "cancelled", "not_found"}:
                message = status.error_message or status.status
                raise HMSVendorError(f"Operation {operation_id} ended with status={status.status}: {message}")
            if time.monotonic() >= deadline:
                raise HMSVendorError(f"Timed out waiting for operation {operation_id}. Last status={status.status}.")
            time.sleep(poll_interval)

    def wait_for_retain(
        self,
        bank_id: str,
        retain_summary: RetainSummary,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ) -> list[OperationStatus]:
        operation_ids = retain_summary.operation_ids or ([retain_summary.operation_id] if retain_summary.operation_id else [])
        if not operation_ids:
            return []
        return [
            self.wait_for_operation(
                bank_id=bank_id,
                operation_id=operation_id,
                timeout=timeout,
                poll_interval=poll_interval,
            )
            for operation_id in operation_ids
        ]

    def _session_to_item(self, session: SessionRecord | dict[str, Any]) -> dict[str, Any]:
        record = self._coerce_session(session)
        metadata = dict(record.metadata)
        metadata.setdefault("source", "vendor_sdk")
        metadata.setdefault("session_id", record.session_id)
        return {
            "content": self._render_session_content(record),
            "timestamp": record.timestamp,
            "context": record.context,
            "document_id": record.session_id,
            "metadata": metadata,
            "tags": record.tags,
            "update_mode": record.update_mode,
            "entities": record.entities,
            "observation_scopes": record.observation_scopes,
            "strategy": record.strategy,
        }

    def _coerce_session(self, session: SessionRecord | dict[str, Any]) -> SessionRecord:
        if isinstance(session, SessionRecord):
            return session
        return SessionRecord(
            session_id=session["session_id"],
            messages=session["messages"],
            timestamp=session.get("timestamp"),
            context=session.get("context"),
            metadata=session.get("metadata", {}),
            tags=session.get("tags", []),
            update_mode=session.get("update_mode"),
            entities=session.get("entities"),
            observation_scopes=session.get("observation_scopes"),
            strategy=session.get("strategy"),
        )

    def _render_session_content(self, record: SessionRecord) -> str:
        rendered_messages = []
        for raw_message in record.messages:
            message = self._coerce_message(raw_message)
            rendered_messages.append(f"[{message.role.upper()}]\n{message.content.strip()}")
        return "\n\n".join(rendered_messages)

    def _coerce_message(self, message: SessionMessage | dict[str, Any]) -> SessionMessage:
        if isinstance(message, SessionMessage):
            return message
        return SessionMessage(role=message["role"], content=message["content"])

    def _validate_sessions(self, sessions: list[SessionRecord | dict[str, Any]]) -> None:
        if not sessions:
            raise HMSVendorError("retain_sessions() requires at least one session.")
        seen_ids: set[str] = set()
        for raw_session in sessions:
            session = self._coerce_session(raw_session)
            if not session.session_id:
                raise HMSVendorError("Every session requires a non-empty session_id.")
            if session.session_id in seen_ids:
                raise HMSVendorError(f"Duplicate session_id in one retain batch: {session.session_id}")
            seen_ids.add(session.session_id)
            if not session.messages:
                raise HMSVendorError(f"Session {session.session_id} has no messages.")
            for message in session.messages:
                coerced = self._coerce_message(message)
                if not coerced.role or not coerced.content:
                    raise HMSVendorError(f"Session {session.session_id} contains an invalid message.")

    def _stringify_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str):
                result[key] = value
            else:
                result[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return result

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = None
        if payload is not None:
            data = json.dumps(self._normalize_payload(payload)).encode("utf-8")

        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HMSVendorError(f"{method} {url} failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise HMSVendorError(f"{method} {url} failed: {exc.reason}") from exc

    def _normalize_payload(self, payload: Any) -> Any:
        if is_dataclass(payload):
            return {key: self._normalize_payload(value) for key, value in asdict(payload).items() if value is not None}
        if isinstance(payload, dict):
            return {key: self._normalize_payload(value) for key, value in payload.items() if value is not None}
        if isinstance(payload, list):
            return [self._normalize_payload(value) for value in payload]
        return payload
