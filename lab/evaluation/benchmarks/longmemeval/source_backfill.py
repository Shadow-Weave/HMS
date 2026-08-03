"""Query-time recovery of source spans retained in ``documents.original_text``.

Fact extraction is intentionally lossy.  When a retrieved fact points to a
document, the original retained transcript is still a trustworthy source for
details that were not materialized as a fact.  This module selects small,
query-focused turn windows; it never invents text or broadens a bank scope.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "about",
        "after",
        "again",
        "and",
        "am",
        "are",
        "as",
        "at",
        "be",
        "before",
        "between",
        "by",
        "can",
        "current",
        "currently",
        "could",
        "different",
        "did",
        "do",
        "does",
        "during",
        "for",
        "first",
        "from",
        "have",
        "has",
        "had",
        "how",
        "i",
        "in",
        "interesting",
        "is",
        "it",
        "me",
        "many",
        "much",
        "might",
        "my",
        "of",
        "on",
        "or",
        "previous",
        "please",
        "recommend",
        "recommendation",
        "recommendations",
        "recent",
        "recently",
        "some",
        "since",
        "should",
        "suggest",
        "suggestion",
        "suggestions",
        "that",
        "the",
        "then",
        "there",
        "this",
        "total",
        "tell",
        "to",
        "find",
        "looking",
        "like",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "with",
        "who",
        "would",
        "you",
        "your",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")
_SIGNAL_RE = re.compile(
    r"(?:\$\s?\d|\b\d+(?:[.,]\d+)?%?|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|"
    r"\b(?:before|after|earlier|later|first|last|current|latest|total|spent|cost|discount)\b)",
    re.IGNORECASE,
)
_QUERY_SIGNAL_RE = re.compile(
    r"\b(?:number|amount|cost|price|date|day|days|week|weeks|month|time|hour|hours|"
    r"how many|how much|earliest|latest|first|last|before|after|current|total|difference|order)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceSnippet:
    document_id: str
    turn_start: int
    turn_end: int
    text: str
    score: int
    focus_turn: int | None = None


@dataclass(frozen=True)
class SourceChunkSnippet:
    """A chunk referenced by a retrieved fact but omitted from the response."""

    chunk_id: str
    document_id: str | None
    chunk_index: int | None
    text: str
    score: int


def _query_terms(query: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_RE.findall(query.lower()) if token not in _STOPWORDS)


def _turn_text(turn: Any) -> str:
    if isinstance(turn, Mapping):
        content = turn.get("content")
        if isinstance(content, str):
            text = content
        elif content is not None:
            text = json.dumps(content, ensure_ascii=False)
        else:
            text = ""
        role = str(turn.get("role") or "").strip().lower()
        return f"{role}: {text}" if role and text else text
    return str(turn or "")


def _parse_turns(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return [raw] if raw.strip() else []
    if isinstance(raw, Mapping):
        raw = raw.get("messages") or raw.get("turns") or [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [_turn_text(turn) for turn in raw if _turn_text(turn).strip()]


def _compact(text: str, limit: int) -> str:
    text = " ".join(text.replace("<|endoftext|>", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalise_for_match(text: Any) -> str:
    """Normalise text for source-coverage and duplicate checks."""

    return re.sub(r"\W+", " ", str(text or "").lower()).strip()


def _content_without_role(text: str) -> str:
    if re.match(r"^(?:user|assistant|system|tool):\s", text, re.IGNORECASE):
        return text.split(":", 1)[1].lstrip()
    return text


def _is_turn_covered(turn: str, covered_chunks: Sequence[str]) -> bool:
    """Return whether a retained turn is already represented by a chunk.

    Recall chunks are JSON windows while source turns are rendered as plain
    text.  Suppress a backfill only when a substantial normalised substring is
    visibly present in an existing chunk.
    """

    if not covered_chunks:
        return False
    normalised_turn = _normalise_for_match(_content_without_role(turn))
    if not normalised_turn:
        return True
    prefix = normalised_turn[:180] if len(normalised_turn) >= 180 else normalised_turn
    for chunk in covered_chunks:
        normalised_chunk = _normalise_for_match(chunk)
        if not normalised_chunk:
            continue
        if normalised_turn in normalised_chunk:
            return True
        # Short turns are commonly embedded intact in a larger chunk.  Do not
        # treat a long-turn prefix as coverage: token-budget truncation can
        # leave the opening sentence while dropping the answer-bearing tail.
        if len(normalised_turn) <= 240 and len(prefix) >= 80 and prefix in normalised_chunk:
            return True
    return False


def _query_cluster_center(text: str, query_terms: frozenset[str], window: int) -> int | None:
    """Find the densest deterministic cluster of query-term matches."""

    matches = sorted(
        (match.start(), match.end(), term)
        for term in query_terms
        for match in re.finditer(rf"\b{re.escape(term)}\b", text)
    )
    if not matches:
        return None

    left = 0
    counts: Counter[str] = Counter()
    best_score: tuple[int, int, int, int] | None = None
    best_bounds = (matches[0][0], matches[0][1])
    for right, (right_position, right_end, term) in enumerate(matches):
        counts[term] += 1
        while left < right and right_end - matches[left][0] > window:
            left_term = matches[left][2]
            counts[left_term] -= 1
            if not counts[left_term]:
                del counts[left_term]
            left += 1
        left_position = matches[left][0]
        span = right_end - left_position
        score = (len(counts), right - left + 1, -span, right_position)
        if best_score is None or score > best_score:
            best_score = score
            best_bounds = (left_position, right_end)
    return sum(best_bounds) // 2


def _compact_relevant(text: str, limit: int, query_terms: frozenset[str]) -> str:
    """Compact one turn around a relevant match instead of from the left."""

    text = " ".join(text.replace("<|endoftext|>", " ").split())
    if len(text) <= limit:
        return text
    role_match = re.match(r"^(?:user|assistant|system|tool):\s", text, re.IGNORECASE)
    role_prefix = role_match.group(0) if role_match else ""
    body = text[len(role_prefix) :]
    if limit <= len(role_prefix) + 6:
        return text[:limit]

    window = limit - len(role_prefix) - 6
    center = _query_cluster_center(body.lower(), query_terms, window)
    if center is None:
        signal_match = _SIGNAL_RE.search(body)
        center = signal_match.start() if signal_match else 0
    start = max(0, center - window // 2)
    if start + window > len(body):
        start = max(0, len(body) - window)
    excerpt = body[start : start + window].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if start + window < len(body):
        excerpt = excerpt.rstrip() + "..."
    return role_prefix + excerpt


def _render_turn_window(
    turns: Sequence[str],
    start: int,
    end: int,
    focus: int,
    query_terms: frozenset[str],
    limit: int,
) -> str:
    """Render the focus turn first, then bounded neighbouring turns."""

    indexes = list(range(start, end))
    if not indexes:
        return ""
    focus_limit = min(limit, max(640, int(limit * 0.62)))
    remaining = max(0, limit - focus_limit)
    neighbours = [index for index in indexes if index != focus]
    neighbour_limit = remaining // len(neighbours) if neighbours else 0
    rendered: list[str] = []
    for index in indexes:
        per_turn = focus_limit if index == focus else neighbour_limit
        if per_turn <= 0:
            continue
        piece = _compact_relevant(turns[index], per_turn, query_terms)
        if piece:
            rendered.append(piece)
    return "\n".join(rendered)


def select_source_snippets(
    documents: Mapping[str, Any],
    query: str,
    document_order: Sequence[str],
    *,
    max_documents: int = 12,
    max_snippets: int = 16,
    max_snippets_per_document: int = 2,
    max_chars_per_snippet: int = 1800,
    max_total_chars: int = 18_000,
    min_score: int = 2,
    covered_chunks: Mapping[str, Sequence[str]] | None = None,
) -> list[SourceSnippet]:
    """Select deterministic source windows from already-retrieved documents.

    Documents are considered in retrieval order.  Within each document, a
    scored turn and its immediate neighbours form one window, preserving the
    conversational context around a matching answer. Existing returned chunks
    can be supplied through ``covered_chunks``; a focus turn that is already
    visible is omitted so this function only adds provenance recall lost.
    Round-robin selection prevents one long transcript from consuming the
    budget.
    """

    terms = _query_terms(query)
    candidates_by_doc: dict[str, list[SourceSnippet]] = {}
    seen_docs: set[str] = set()
    for document_id in document_order[:max_documents]:
        document_id = str(document_id)
        if not document_id or document_id in seen_docs or document_id not in documents:
            continue
        seen_docs.add(document_id)
        turns = _parse_turns(documents[document_id])
        if not turns:
            continue

        scored_turns: list[tuple[int, int, int]] = []
        has_covered_overlap = False
        for index, turn in enumerate(turns):
            lower = turn.lower()
            overlap = sum(term in lower for term in terms)
            signal = 1 if _SIGNAL_RE.search(turn) else 0
            score = overlap * 5 + signal * 2
            if score < min_score:
                continue
            if covered_chunks and _is_turn_covered(turn, covered_chunks.get(document_id, ())):
                has_covered_overlap = has_covered_overlap or overlap > 0
                continue
            scored_turns.append((score, overlap, index))
        if any(overlap > 0 for _, overlap, _ in scored_turns):
            scored_turns = [row for row in scored_turns if row[1] > 0]
        elif has_covered_overlap or not _QUERY_SIGNAL_RE.search(query):
            # Query-relevant evidence for this document is already visible;
            # do not add an unrelated numeric/date-only fallback window.  A
            # signal-only fallback remains available for explicitly numeric or
            # temporal questions such as "what was the amount?".
            scored_turns = []
        scored_turns.sort(key=lambda row: (-row[0], -row[1], row[2]))

        snippets: list[SourceSnippet] = []
        seen_windows: set[str] = set()
        for score, _, index in scored_turns:
            start = max(0, index - 1)
            end = min(len(turns), index + 2)
            text = _render_turn_window(turns, start, end, index, terms, max_chars_per_snippet)
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if not text or key in seen_windows:
                continue
            seen_windows.add(key)
            snippets.append(SourceSnippet(document_id, start, end, text, score, index))
            if len(snippets) >= max_snippets_per_document:
                break
        if snippets:
            candidates_by_doc[document_id] = snippets

    selected: list[SourceSnippet] = []
    total_chars = 0
    # Round-robin gives each high-ranked source a chance before a second window
    # from any one document is added.
    for offset in range(max_snippets_per_document):
        for document_id in document_order[:max_documents]:
            snippets = candidates_by_doc.get(str(document_id), [])
            if offset >= len(snippets):
                continue
            snippet = snippets[offset]
            if len(selected) >= max_snippets or total_chars + len(snippet.text) > max_total_chars:
                return selected
            selected.append(snippet)
            total_chars += len(snippet.text)
    return selected


def select_missing_chunk_snippets(
    chunks: Sequence[Mapping[str, Any]],
    query: str,
    *,
    max_chunks: int = 6,
    max_chars_per_chunk: int = 1200,
    max_total_chars: int = 7_000,
) -> list[SourceChunkSnippet]:
    """Select exact chunk rows that recall referenced but did not return.

    This is the highest-fidelity recovery path: the chunk was already linked
    to a retrieved fact, so no bank-wide search or new extraction is needed.
    Rows are ranked by query overlap and retain input order as a stable tie
    breaker (the caller supplies fact/retrieval order).
    """

    terms = _query_terms(query)
    candidates: list[tuple[int, int, SourceChunkSnippet]] = []
    for position, row in enumerate(chunks):
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        raw_text = row.get("chunk_text") or ""
        turns = _parse_turns(raw_text)
        best_score = 0
        best_index = 0
        retrieval_rank = int(row.get("_retrieval_rank") or position + 1)
        rank_bonus = max(0, 12 - min(retrieval_rank, 12))
        if turns:
            turn_scores = []
            for index, turn in enumerate(turns):
                overlap = sum(term in turn.lower() for term in terms)
                signal = 1 if _SIGNAL_RE.search(turn) else 0
                turn_scores.append((overlap * 5 + signal * 2, overlap, index))
            best_score, _, best_index = max(turn_scores, key=lambda item: (item[0], item[1], -item[2]))
            text = _render_turn_window(
                turns,
                max(0, best_index - 1),
                min(len(turns), best_index + 2),
                best_index,
                terms,
                max_chars_per_chunk,
            )
        else:
            text = _compact_relevant(str(raw_text), max_chars_per_chunk, terms)
            best_score = sum(term in str(raw_text).lower() for term in terms) * 5
            if _SIGNAL_RE.search(str(raw_text)):
                best_score += 2
        best_score += rank_bonus
        if not text:
            continue
        candidates.append(
            (
                -best_score,
                position,
                SourceChunkSnippet(
                    chunk_id=chunk_id,
                    document_id=str(row.get("document_id") or "") or None,
                    chunk_index=row.get("chunk_index"),
                    text=text,
                    score=best_score,
                ),
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[SourceChunkSnippet] = []
    seen_ids: set[str] = set()
    total_chars = 0
    for _, _, candidate in candidates:
        if candidate.chunk_id in seen_ids:
            continue
        if len(selected) >= max_chunks or total_chars + len(candidate.text) > max_total_chars:
            break
        seen_ids.add(candidate.chunk_id)
        selected.append(candidate)
        total_chars += len(candidate.text)
    return selected


def render_source_chunk_snippets(snippets: Sequence[SourceChunkSnippet]) -> str:
    """Render exact chunk recoveries with explicit chunk provenance."""

    if not snippets:
        return ""
    lines = [
        "=== Retrieved-Chunk Provenance Recovery ===",
        "These are exact retained chunks linked to retrieved facts but omitted from the normal response because of the chunk token budget.",
        "",
    ]
    for index, snippet in enumerate(snippets, 1):
        location = f"document={snippet.document_id or '-'} | chunk_index={snippet.chunk_index if snippet.chunk_index is not None else '-'}"
        lines.append(
            f"{index}. chunk={snippet.chunk_id} | {location} | relevance_score={snippet.score} | {snippet.text}"
        )
    return "\n".join(lines)


def render_source_snippets(snippets: Sequence[SourceSnippet]) -> str:
    """Render source snippets with explicit provenance and no synthetic facts."""

    if not snippets:
        return ""
    lines = [
        "=== Retained Source-Document Evidence ===",
        "These excerpts are verbatim windows from retained documents selected because their source document was retrieved. They are provenance evidence, not new inferred facts.",
        "",
    ]
    for index, snippet in enumerate(snippets, 1):
        lines.append(
            f"{index}. document={snippet.document_id} | turns={snippet.turn_start}-{snippet.turn_end - 1} | "
            f"relevance_score={snippet.score} | {snippet.text}"
        )
    return "\n".join(lines)
