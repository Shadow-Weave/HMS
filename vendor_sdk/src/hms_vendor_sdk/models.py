from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SessionMessage:
    role: str
    content: str


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    messages: list[SessionMessage | dict[str, Any]]
    timestamp: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    update_mode: str | None = None
    entities: list[dict[str, str]] | None = None
    observation_scopes: str | list[list[str]] | None = None
    strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetainSummary:
    bank_id: str
    items_count: int
    success: bool
    async_mode: bool
    operation_id: str | None = None
    operation_ids: list[str] | None = None
    usage: dict[str, Any] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OperationStatus:
    operation_id: str
    status: str
    operation_type: str | None = None
    error_message: str | None = None
    result_metadata: dict[str, Any] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecallItem:
    id: str
    text: str
    type: str | None = None
    entities: list[str] | None = None
    context: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None
    mentioned_at: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] | None = None
    chunk_id: str | None = None
    tags: list[str] | None = None
    source_fact_ids: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RecallItem":
        known_fields = {
            "id",
            "text",
            "type",
            "entities",
            "context",
            "occurred_start",
            "occurred_end",
            "mentioned_at",
            "document_id",
            "metadata",
            "chunk_id",
            "tags",
            "source_fact_ids",
        }
        known = {key: row.get(key) for key in known_fields if key in row}
        extra = {key: value for key, value in row.items() if key not in known_fields}
        return cls(extra=extra, **known)


@dataclass(slots=True)
class RecallBundle:
    bank_id: str
    question: str
    question_date: str | None
    results: list[RecallItem]
    trace: dict[str, Any] | None = None
    entities: dict[str, Any] | None = None
    chunks: dict[str, Any] | None = None
    source_facts: dict[str, Any] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "question": self.question,
            "question_date": self.question_date,
            "results": [item.to_dict() for item in self.results],
            "trace": self.trace,
            "entities": self.entities,
            "chunks": self.chunks,
            "source_facts": self.source_facts,
            "raw_response": self.raw_response,
        }


@dataclass(slots=True)
class EvidenceLedgerRow:
    index: int
    score: int
    text: str
    document_id: str | None = None
    type: str | None = None
    occurred: str | None = None
    mentioned: str | None = None
    chunk_id: str | None = None
    entities: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceControl:
    name: str
    triggered: bool
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidencePacket:
    question: str
    question_date: str | None
    mode: str
    ledger_rows: list[EvidenceLedgerRow]
    controls: list[EvidenceControl]
    source_snippets: list[dict[str, Any]]
    answer_ready_context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "question_date": self.question_date,
            "mode": self.mode,
            "ledger_rows": [row.to_dict() for row in self.ledger_rows],
            "controls": [control.to_dict() for control in self.controls],
            "source_snippets": self.source_snippets,
            "answer_ready_context": self.answer_ready_context,
        }


@dataclass(slots=True)
class PipelineResult:
    bank_id: str
    retain_summary: RetainSummary
    recall_bundle: RecallBundle
    evidence_packet: EvidencePacket | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "retain_summary": self.retain_summary.to_dict(),
            "recall_bundle": self.recall_bundle.to_dict(),
            "evidence_packet": self.evidence_packet.to_dict() if self.evidence_packet else None,
        }


@dataclass(slots=True)
class VendorCase:
    case_id: str
    question: str
    sessions: list[SessionRecord | dict[str, Any]]
    question_date: str | None = None
    bank_profile: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
