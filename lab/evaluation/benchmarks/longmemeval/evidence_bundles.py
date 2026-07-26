"""Source-centric evidence selection for LongMemEval answer prompts.

The recall API returns facts, while several facts can point at the same raw
source chunk. Rendering that relationship as ``fact -> chunk`` for every fact
inflates the prompt and makes repeated wording look like independent events.
This module keeps the retrieval order and provenance, but renders each source
chunk once with a small, query-focused set of facts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_STOPWORDS = frozenset(
    {
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
)
_SIGNAL_RE = re.compile(
    r"(?:\$\s?\d|\b\d+(?:[.,]\d+)?%?|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|"
    r"\b(?:before|after|earlier|later|first|last|current|latest|total|spent|cost|discount)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceBundle:
    """Facts sharing one source, in the order in which the source was found."""

    source_key: str
    document_id: str | None
    chunk_id: str | None
    first_rank: int
    facts: tuple[dict[str, Any], ...]
    chunk: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RenderedEvidence:
    """Rendered prompt text and the source excerpts actually visible in it."""

    text: str
    covered_by_document: dict[str, tuple[str, ...]]


def _terms(query: str) -> frozenset[str]:
    return frozenset(
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", query.lower()) if token not in _STOPWORDS
    )


def _normalise_text(value: Any) -> str:
    return re.sub(r"\W+", " ", str(value or "").lower()).strip()


def _compact_chunk_text(text: str, query_terms: frozenset[str], limit: int) -> str:
    """Keep query-relevant turns when a raw chunk exceeds the display budget."""

    text = " ".join(str(text or "").replace("<|endoftext|>", " ").split())
    if len(text) <= limit:
        return text

    turns: list[str] = []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        parsed = parsed.get("messages") or parsed.get("turns") or [parsed]
    if isinstance(parsed, list):
        for turn in parsed:
            if isinstance(turn, Mapping):
                content = turn.get("content")
                if content is None:
                    continue
                role = str(turn.get("role") or "").strip().lower()
                body = str(content)
                turns.append(f"{role}: {body}" if role else body)
            elif turn:
                turns.append(str(turn))
    if not turns:
        head = max(1, (limit - 9) // 2)
        tail = max(1, limit - 9 - head)
        return f"{text[:head].rstrip()} ... {text[-tail:].lstrip()}"

    scores = [sum(term in turn.lower() for term in query_terms) for turn in turns]
    focus = max(range(len(turns)), key=lambda index: (scores[index], -index))
    focus_limit = min(limit, max(480, int(limit * 0.62)))
    remaining = max(0, limit - focus_limit)
    neighbours = [index for index in range(len(turns)) if index != focus]
    neighbour_limit = remaining // len(neighbours) if neighbours else 0
    pieces: list[str] = []
    for index, turn in enumerate(turns):
        per_turn = focus_limit if index == focus else neighbour_limit
        if per_turn <= 0:
            continue
        clean = " ".join(turn.split())
        if len(clean) > per_turn:
            original_length = len(clean)
            lower = clean.lower()
            positions = [
                match.start() for term in query_terms for match in re.finditer(rf"\b{re.escape(term)}\b", lower)
            ]
            # The last match often carries the updated state in a long
            # assistant turn, while retaining the whole neighbouring turn
            # keeps the user assertion visible.
            center = max(positions, default=0)
            window = max(1, per_turn - 6)
            start = max(0, min(center - window // 2, original_length - window))
            clean = ("..." if start else "") + clean[start : start + window]
            if start + window < original_length:
                clean += "..."
        pieces.append(clean)
    return "\n".join(pieces)[:limit]


def _priority(fact: Mapping[str, Any], query_terms: frozenset[str], rank: int) -> tuple[int, int]:
    text = str(fact.get("text") or "")
    lower = text.lower()
    overlap = sum(term in lower for term in query_terms)
    signal = 1 if _SIGNAL_RE.search(text) else 0
    # Relevance wins within a source; the original rank is a stable tie-breaker.
    return (overlap * 4 + signal * 2, -rank)


def build_evidence_bundles(
    results: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, Mapping[str, Any]] | None,
    query: str,
    *,
    max_bundles: int = 96,
    max_facts_per_bundle: int = 2,
) -> list[EvidenceBundle]:
    """Group retrieved facts by source chunk without changing candidate recall.

    ``results`` is assumed to be in retrieval order.  Facts without a chunk
    remain addressable by document (or their own id), so observations and
    source-less rows are not silently discarded.  The function is deterministic and
    has no database or model dependency, which makes it suitable for both
    benchmark prompts and unit tests.
    """

    if max_bundles <= 0 or max_facts_per_bundle <= 0:
        return []

    query_terms = _terms(query)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    order: list[str] = []
    for rank, fact in enumerate(results, 1):
        if not isinstance(fact, Mapping):
            continue
        chunk_id = str(fact.get("chunk_id") or "") or None
        document_id = str(fact.get("document_id") or "") or None
        source_key = chunk_id or document_id or f"fact:{fact.get('id') or rank}"
        if source_key not in grouped:
            grouped[source_key] = []
            order.append(source_key)
        grouped[source_key].append((rank, fact))

    bundles: list[EvidenceBundle] = []
    chunks = chunks or {}
    for source_key in order[:max_bundles]:
        rows = grouped[source_key]
        first_rank = rows[0][0]
        # Avoid repeated extraction of the same sentence within a chunk while
        # keeping distinct numeric/event facts available to the answer model.
        seen_text: set[str] = set()
        ranked_rows = sorted(
            rows,
            key=lambda row: _priority(row[1], query_terms, row[0]),
            reverse=True,
        )
        selected: list[tuple[int, Mapping[str, Any]]] = []
        for rank, fact in ranked_rows:
            text_key = _normalise_text(fact.get("text"))
            if text_key and text_key in seen_text:
                continue
            if text_key:
                seen_text.add(text_key)
            selected.append((rank, fact))
            if len(selected) >= max_facts_per_bundle:
                break

        if not selected:
            continue
        selected.sort(key=lambda row: row[0])
        first_fact = selected[0][1]
        chunk_id = str(first_fact.get("chunk_id") or "") or None
        document_id = str(first_fact.get("document_id") or "") or None
        chunk = chunks.get(chunk_id) if chunk_id else None
        bundles.append(
            EvidenceBundle(
                source_key=source_key,
                document_id=document_id,
                chunk_id=chunk_id,
                first_rank=first_rank,
                facts=tuple(dict(fact) for _, fact in selected),
                chunk=chunk,
            )
        )

    bundles.sort(key=lambda bundle: bundle.first_rank)
    return bundles


def render_evidence_with_coverage(
    bundles: Sequence[EvidenceBundle],
    *,
    max_chunk_chars: int = 1400,
    max_total_chars: int | None = None,
    query: str = "",
) -> RenderedEvidence:
    """Render bundles with one raw source chunk per bundle.

    ``recall`` can return hundreds of candidates.  A source-centric layout
    removes duplicate chunks, but it still needs an explicit prompt budget;
    otherwise one chunk per candidate can exceed the answer model's useful
    attention window.  When ``max_total_chars`` is set, whole bundles are
    retained in retrieval order until the bound is reached.
    """

    if not bundles:
        return RenderedEvidence(text="", covered_by_document={})

    header = "\n".join(
        [
            "=== Source-Centric Evidence Bundles ===",
            "Each source chunk is shown once. Facts under the same source are not independent events; deduplicate them before counting.",
        ]
    )
    if max_total_chars is not None and len(header) > max_total_chars:
        return RenderedEvidence(text="", covered_by_document={})

    rendered_text = header
    covered_by_document: dict[str, list[str]] = {}
    for index, bundle in enumerate(bundles, 1):
        source = bundle.chunk_id or bundle.document_id or bundle.source_key
        bundle_lines = [f"Bundle {index} (retrieval_rank={bundle.first_rank}, source={source}):"]
        for fact in bundle.facts:
            when_parts = []
            if fact.get("occurred_start"):
                when_parts.append(f"occurred={fact['occurred_start']}")
            if fact.get("occurred_end") and fact.get("occurred_end") != fact.get("occurred_start"):
                when_parts.append(f"ended={fact['occurred_end']}")
            if fact.get("mentioned_at"):
                when_parts.append(f"mentioned={fact['mentioned_at']}")
            when = ", ".join(when_parts) or "unknown time"
            bundle_lines.append(f"- Fact ({fact.get('fact_type', 'unknown')}, {when}): {fact.get('text', '')}")
        chunk_text = ""
        if bundle.chunk:
            chunk_text = _compact_chunk_text(str(bundle.chunk.get("chunk_text") or ""), _terms(query), max_chunk_chars)
            if chunk_text:
                bundle_lines.append(f'- Source chunk: "{chunk_text}"')
        bundle_text = "\n".join(bundle_lines)
        candidate_text = f"{rendered_text}\n\n{bundle_text}"
        if max_total_chars is not None and len(candidate_text) > max_total_chars:
            break
        rendered_text = candidate_text
        if bundle.document_id and chunk_text:
            covered_by_document.setdefault(bundle.document_id, []).append(chunk_text)

    return RenderedEvidence(
        text=rendered_text,
        covered_by_document={document_id: tuple(excerpts) for document_id, excerpts in covered_by_document.items()},
    )


def render_evidence_bundles(
    bundles: Sequence[EvidenceBundle],
    *,
    max_chunk_chars: int = 1400,
    max_total_chars: int | None = None,
    query: str = "",
) -> str:
    """Render evidence as text, preserving the original helper contract.

    Call :func:`render_evidence_with_coverage` when the caller also needs the
    exact excerpts admitted by the prompt budget.
    """

    return render_evidence_with_coverage(
        bundles,
        max_chunk_chars=max_chunk_chars,
        max_total_chars=max_total_chars,
        query=query,
    ).text
