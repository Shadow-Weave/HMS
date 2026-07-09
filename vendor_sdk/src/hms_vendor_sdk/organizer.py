from __future__ import annotations

import re
from typing import Any

from .models import EvidenceControl, EvidenceLedgerRow, EvidencePacket, RecallBundle, RecallItem


SIGNAL_RE = re.compile(
    r"(\$?\d+(?:[.,]\d+)?%?|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|yesterday|tomorrow|last|next|ago|week|month|year|day|hour|"
    r"before|after|first|earlier|later|previous|current|latest|total|spent|cost|discount|cashback)",
    re.IGNORECASE,
)

STOPWORDS = {
    "about",
    "after",
    "again",
    "before",
    "between",
    "current",
    "currently",
    "different",
    "during",
    "first",
    "from",
    "have",
    "many",
    "much",
    "previous",
    "recently",
    "since",
    "that",
    "the",
    "then",
    "there",
    "this",
    "total",
    "what",
    "when",
    "where",
    "which",
    "with",
}


class EvidenceOrganizer:
    """Answer-time evidence organization extracted from the benchmark pipeline."""

    def organize(
        self,
        question: str,
        recall_bundle: RecallBundle,
        *,
        question_date: str | None = None,
        mode: str = "structured_ledger",
        max_rows: int = 45,
        max_source_snippets: int = 18,
    ) -> EvidencePacket:
        rows = self._build_rows(question, recall_bundle.results, max_rows=max_rows)
        snippets = self._source_snippets(rows, recall_bundle.chunks or {}, max_source_snippets=max_source_snippets)
        controls = self._controls(question)
        context = self._answer_ready_context(question, question_date, rows, controls, snippets)
        return EvidencePacket(
            question=question,
            question_date=question_date,
            mode=mode,
            ledger_rows=rows,
            controls=controls,
            source_snippets=snippets,
            answer_ready_context=context,
        )

    def _build_rows(self, question: str, results: list[RecallItem], *, max_rows: int) -> list[EvidenceLedgerRow]:
        question_terms = self._content_terms(question)
        rows: list[EvidenceLedgerRow] = []
        seen: set[str] = set()
        for fact in results[:180]:
            text = self._compact_text(fact.text, 360)
            if not text:
                continue
            text_lower = text.lower()
            term_overlap = sum(1 for term in question_terms if term in text_lower)
            has_signal = bool(SIGNAL_RE.search(text))
            if not has_signal and term_overlap < 2:
                continue

            doc_id = str(fact.document_id or "")
            dedupe_key = re.sub(r"\W+", " ", text_lower).strip()[:180]
            dedupe_key = f"{doc_id}:{dedupe_key}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            rows.append(
                EvidenceLedgerRow(
                    index=0,
                    score=(3 if has_signal else 0) + term_overlap,
                    text=text,
                    document_id=fact.document_id,
                    type=fact.type,
                    occurred=fact.occurred_start or fact.occurred_end or None,
                    mentioned=fact.mentioned_at,
                    chunk_id=fact.chunk_id,
                    entities=fact.entities,
                )
            )
            if len(rows) >= 70:
                break

        rows.sort(key=lambda row: row.score, reverse=True)
        rows = rows[:max_rows]
        for index, row in enumerate(rows, start=1):
            row.index = index
        return rows

    def _source_snippets(
        self,
        rows: list[EvidenceLedgerRow],
        chunks: dict[str, Any],
        *,
        max_source_snippets: int,
    ) -> list[dict[str, Any]]:
        snippets: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        for row in rows:
            chunk_id = row.chunk_id
            if not chunk_id or chunk_id in seen_chunks or chunk_id not in chunks:
                continue
            seen_chunks.add(chunk_id)
            chunk_info = chunks.get(chunk_id) or {}
            text = chunk_info.get("text") or chunk_info.get("chunk_text") or ""
            if text:
                snippets.append(
                    {
                        "chunk_id": chunk_id,
                        "text": self._compact_text(text, 650),
                        "ledger_row": row.index,
                    }
                )
            if len(snippets) >= max_source_snippets:
                break
        return snippets

    def _controls(self, question: str) -> list[EvidenceControl]:
        q = question.lower()
        controls = [
            EvidenceControl(
                name="count_total_deduplication",
                triggered=any(marker in q for marker in ("how many", "in total", "total", "count")),
                instruction=(
                    "Enumerate unique real user events/items before counting. Do not count duplicate mentions, "
                    "recommendations, options, or generic background facts."
                ),
            ),
            EvidenceControl(
                name="relative_date_grounding",
                triggered=any(marker in q for marker in ("ago", "before", "after", "last ", "next ", "yesterday", "tomorrow")),
                instruction=(
                    "Resolve relative dates against question_date, then prefer facts whose event time or source text "
                    "matches the resolved time window."
                ),
            ),
            EvidenceControl(
                name="amount_difference_calibration",
                triggered=any(marker in q for marker in ("how much", "difference", "more", "less", "cost", "spent", "amount")),
                instruction=(
                    "Compute amount/difference only when both requested sides are explicitly present. If a side is "
                    "missing, state what is missing instead of treating it as zero."
                ),
            ),
            EvidenceControl(
                name="current_previous_state_arbitration",
                triggered=any(marker in q for marker in ("current", "currently", "latest", "previous", "initially", "before")),
                instruction=(
                    "For current-state questions, prefer the latest explicit state. For previous/before questions, "
                    "prefer the older explicit state. Do not add old and new states unless the question asks for a cumulative total."
                ),
            ),
        ]
        return controls

    def _answer_ready_context(
        self,
        question: str,
        question_date: str | None,
        rows: list[EvidenceLedgerRow],
        controls: list[EvidenceControl],
        snippets: list[dict[str, Any]],
    ) -> str:
        lines = [
            "=== Structured Evidence Ledger ===",
            "Use this as an answer-time checklist for count/sum/date/order/update questions. It is extracted from retrieved memory evidence; do not treat it as new evidence. Deduplicate repeated mentions of the same event before counting.",
            f"Question: {question}",
            f"Question date: {question_date or 'not specified'}",
            "",
            "Active controls:",
        ]
        active_controls = [control for control in controls if control.triggered]
        if active_controls:
            for control in active_controls:
                lines.append(f"- {control.name}: {control.instruction}")
        else:
            lines.append("- none")

        lines.extend(["", "Candidate facts:"])
        if rows:
            for row in rows:
                lines.append(
                    f"{row.index}. occurred={row.occurred or '-'} | mentioned={row.mentioned or '-'} | "
                    f"doc={row.document_id or '-'} | type={row.type or '-'} | {row.text}"
                )
        else:
            lines.append("- no ledger rows passed filtering")

        if snippets:
            lines.extend(["", "Raw source snippets for ledger rows:"])
            for index, snippet in enumerate(snippets, start=1):
                lines.append(f"{index}. row={snippet['ledger_row']} | chunk={snippet['chunk_id']} | {snippet['text']}")

        return "\n".join(lines)

    @staticmethod
    def _compact_text(text: Any, max_chars: int) -> str:
        compact = " ".join(str(text or "").replace("<|endoftext|>", " ").split())
        if len(compact) > max_chars:
            compact = compact[: max_chars - 3].rstrip() + "..."
        return compact

    @staticmethod
    def _content_terms(question: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", question.lower())
            if token not in STOPWORDS
        }
