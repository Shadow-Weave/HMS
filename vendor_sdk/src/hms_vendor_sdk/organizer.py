from __future__ import annotations

from .models import EvidenceLedgerRow, EvidencePacket, RecallBundle


class EvidenceOrganizer:
    """Serialize recalled rows without filtering, scoring, or answer rules."""

    def organize(
        self,
        question: str,
        recall_bundle: RecallBundle,
        *,
        question_date: str | None = None,
        mode: str = "ordered_recall",
    ) -> EvidencePacket:
        rows = [
            EvidenceLedgerRow(
                index=index,
                score=0,
                text=str(item.text or ""),
                document_id=item.document_id,
                type=item.type,
                occurred=item.occurred_start or item.occurred_end,
                mentioned=item.mentioned_at,
                chunk_id=item.chunk_id,
                entities=item.entities,
            )
            for index, item in enumerate(recall_bundle.results, start=1)
        ]

        lines = [
            "=== Retrieved Memory Evidence ===",
            f"Question: {question}",
            f"Question date: {question_date or 'not specified'}",
            "",
            "Recalled facts (original order):",
        ]
        if rows:
            lines.extend(
                f"{row.index}. occurred={row.occurred or '-'} | mentioned={row.mentioned or '-'} | "
                f"doc={row.document_id or '-'} | type={row.type or '-'} | {row.text}"
                for row in rows
            )
        else:
            lines.append("- none")

        return EvidencePacket(
            question=question,
            question_date=question_date,
            mode=mode,
            ledger_rows=rows,
            controls=[],
            source_snippets=[],
            answer_ready_context="\n".join(lines),
        )
