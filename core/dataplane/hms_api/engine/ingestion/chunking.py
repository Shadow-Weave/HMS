"""Deterministic, side-effect-free chunk planning for Retain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from .domain import ChunkPlan, ChunkPolicy, ContentItem

PLAIN_TEXT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
)


def compute_content_hash(text: str) -> str:
    """Return the lowercase SHA-256 digest of the text's UTF-8 bytes."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_text(text: str, policy: ChunkPolicy) -> tuple[str, ...]:
    """Split one content item according to the versioned chunk policy.

    The default behavior follows the Retain chunking contract:

    * text at or below ``max_chars`` is returned byte-for-byte;
    * a JSON array containing only objects is treated as a conversation and
      split only between complete turns;
    * all other text uses ``RecursiveCharacterTextSplitter`` with the
      configured ordered separator list.

    A single conversation turn is never split, even if it is larger than the
    configured limit. Conversation overlap is deliberately unsupported because
    repeating turns would change Retain's fact semantics; callers can disable
    conversation mode when character overlap is required.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    # Empty text is represented by one empty chunk.
    if len(text) <= policy.max_chars:
        return (text,)

    if policy.conversation_mode:
        turns = _parse_conversation(text)
        if turns is not None:
            if policy.overlap:
                raise ValueError("conversation chunking does not support overlap")
            return _split_conversation(turns, policy.max_chars)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=policy.max_chars,
        chunk_overlap=policy.overlap,
        length_function=len,
        is_separator_regex=False,
        separators=list(PLAIN_TEXT_SEPARATORS),
    )
    return tuple(splitter.split_text(text))


def build_chunk_plans(
    document_id: str,
    items: Sequence[ContentItem],
    policy: ChunkPolicy,
) -> tuple[ChunkPlan, ...]:
    """Build stable chunk plans for all items in one document.

    ``local_index`` restarts for every content item while ``global_index`` is
    assigned synchronously across the complete document. Consequently neither
    index depends on task completion order. ``chunk_key`` is an unambiguous,
    deterministic composition of document ID, global index, and content hash;
    it never relies on Python's process-randomized :func:`hash`.
    """

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")

    plans: list[ChunkPlan] = []
    global_index = 0
    for item in items:
        for local_index, chunk in enumerate(split_text(item.content, policy)):
            content_hash = compute_content_hash(chunk)
            chunk_key = _build_chunk_key(document_id, global_index, content_hash)
            plans.append(
                ChunkPlan(
                    chunk_key=chunk_key,
                    source_index=item.source_index,
                    global_index=global_index,
                    local_index=local_index,
                    text=chunk,
                    content_hash=content_hash,
                )
            )
            global_index += 1
    return tuple(plans)


def _build_chunk_key(document_id: str, global_index: int, content_hash: str) -> str:
    # Prefixing the document length makes the representation unambiguous even
    # when a caller-supplied document ID contains colons.
    return f"chunk:{len(document_id)}:{document_id}:{global_index}:{content_hash}"


def _parse_conversation(text: str) -> list[dict[str, Any]] | None:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(parsed, list) and all(isinstance(turn, dict) for turn in parsed):
        return parsed
    return None


def _split_conversation(turns: list[dict[str, Any]], max_chars: int) -> tuple[str, ...]:
    chunks: list[str] = []
    current_chunk: list[dict[str, Any]] = []
    current_size = 2  # Serialized []

    for turn in turns:
        turn_json = json.dumps(turn, ensure_ascii=False)
        turn_size = len(turn_json) + 1  # Comma between adjacent turns.

        if current_size + turn_size > max_chars and current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = []
            current_size = 2

        current_chunk.append(turn)
        current_size += turn_size

    if current_chunk:
        chunks.append(json.dumps(current_chunk, ensure_ascii=False))

    # The branch is normally reached only for text larger than max_chars, but
    # retaining the fallback keeps the helper total for an empty JSON array.
    return tuple(chunks) if chunks else (json.dumps(turns, ensure_ascii=False),)
