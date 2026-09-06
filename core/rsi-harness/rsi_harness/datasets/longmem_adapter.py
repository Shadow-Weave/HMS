"""Leakage-aware adapter for LongMemEval-shaped records.

The runtime conversion methods expose only source sessions and the query.  The
reference answer stays in the dataset boundary and is never copied into
``QueryInput`` or ``SessionInput``.  Callers must keep the raw result of
``load`` inside the evaluation boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..session_models import QueryInput, SessionInput


class LongMemEvalItemError(ValueError):
    """Raised when a LongMemEval-shaped item violates its source contract."""


@dataclass(frozen=True, slots=True)
class LongMemEvalRuntimeItem:
    """Sanitized runtime view of one item, with labels excluded."""

    item_id: str
    sessions: tuple[SessionInput, ...]
    query: QueryInput


class LongMemEvalAdapter:
    """Convert one LongMemEval item into chronological runtime inputs."""

    def load(self, path: str | Path, *, max_items: int | None = None) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise LongMemEvalItemError("LongMemEval dataset must be a JSON list")
        items = payload if max_items is None else payload[:max_items]
        for item in items:
            self._validate_item(item)
        return items

    def sessions(
        self,
        item: Mapping[str, Any],
        *,
        stream_id: str | None = None,
        max_sessions: int | None = None,
    ) -> tuple[SessionInput, ...]:
        self._validate_item(item)
        question_id = str(item["question_id"])
        stream = stream_id or f"longmemeval:{question_id}"
        raw_sessions = item["haystack_sessions"]
        raw_dates = item["haystack_dates"]
        raw_ids = item["haystack_session_ids"]
        if max_sessions is not None:
            if max_sessions < 0:
                raise ValueError("max_sessions must be non-negative")
            raw_sessions = raw_sessions[:max_sessions]
            raw_dates = raw_dates[:max_sessions]
            raw_ids = raw_ids[:max_sessions]
        result: list[SessionInput] = []
        seen_ids: set[str] = set()
        for position, (turns, date_value, raw_session_id) in enumerate(
            zip(raw_sessions, raw_dates, raw_ids, strict=True)
        ):
            session_id = str(raw_session_id).strip()
            if not session_id or session_id in seen_ids:
                raise LongMemEvalItemError(f"duplicate or empty session_id: {session_id!r}")
            seen_ids.add(session_id)
            event_time = self._parse_date(date_value)
            messages = self._messages(turns)
            result.append(
                SessionInput(
                    stream_id=stream,
                    session_id=session_id,
                    source_position=position,
                    messages=messages,
                    started_at=event_time,
                    ended_at=event_time,
                    metadata={
                        "dataset": "longmemeval",
                        "question_id": question_id,
                        "source_session_id": session_id,
                        "source_date": str(date_value),
                    },
                )
            )
        return tuple(result)

    def query(
        self,
        item: Mapping[str, Any],
        *,
        evaluation_id: str,
        stream_id: str | None = None,
        query_position: int | None = None,
    ) -> QueryInput:
        self._validate_item(item)
        question_id = str(item["question_id"])
        sessions = item["haystack_session_ids"]
        position = len(sessions) if query_position is None else query_position
        return QueryInput(
            evaluation_id=evaluation_id,
            stream_id=stream_id or f"longmemeval:{question_id}",
            query_id=question_id,
            query=str(item["question"]),
            query_position=position,
            asked_at=self._parse_date(item["question_date"]) if item.get("question_date") else None,
        )

    def runtime(
        self,
        item: Mapping[str, Any],
        *,
        evaluation_id: str,
        stream_id: str | None = None,
        max_sessions: int | None = None,
        query_position: int | None = None,
    ) -> LongMemEvalRuntimeItem:
        """Build the only view allowed to enter the session runtime.

        The returned object contains source sessions and the query, but never
        carries ``answer``, ``question_type`` or ``has_answer`` annotations.
        Keep the raw item inside the evaluation/verifier boundary.
        """

        self._validate_item(item)
        question_id = str(item["question_id"])
        stream = stream_id or f"longmemeval:{question_id}"
        sessions = self.sessions(item, stream_id=stream, max_sessions=max_sessions)
        query = self.query(
            item,
            evaluation_id=evaluation_id,
            stream_id=stream,
            query_position=query_position,
        )
        return LongMemEvalRuntimeItem(question_id, sessions, query)

    @staticmethod
    def _validate_item(item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise LongMemEvalItemError("LongMemEval item must be a mapping")
        question_id = str(item.get("question_id") or "").strip()
        if not question_id:
            raise LongMemEvalItemError("LongMemEval item has no question_id")
        for field in ("haystack_sessions", "haystack_dates", "haystack_session_ids"):
            if not isinstance(item.get(field), list):
                raise LongMemEvalItemError(f"LongMemEval item has no valid {field}")
        lengths = {len(item["haystack_sessions"]), len(item["haystack_dates"]), len(item["haystack_session_ids"])}
        if len(lengths) != 1:
            raise LongMemEvalItemError("LongMemEval session arrays are not aligned")
        if not str(item.get("question") or "").strip():
            raise LongMemEvalItemError("LongMemEval item has no question")

    @staticmethod
    def _messages(turns: Any) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(turns, list) or not turns:
            raise LongMemEvalItemError("LongMemEval session must contain turns")
        messages: list[Mapping[str, Any]] = []
        for turn in turns:
            if isinstance(turn, Mapping):
                content = str(turn.get("content") or "").strip()
                role = str(turn.get("role") or "").strip()
                if not content or not role:
                    raise LongMemEvalItemError("LongMemEval turn must have role and content")
                # ``has_answer`` is a dataset annotation and must not enter
                # the runtime memory/event contract.
                messages.append({"role": role, "content": content})
            elif isinstance(turn, str) and turn.strip():
                messages.append({"role": "unknown", "content": turn.strip()})
            else:
                raise LongMemEvalItemError("LongMemEval turn must be a mapping or non-empty string")
        return tuple(messages)

    @staticmethod
    def _parse_date(value: Any) -> datetime:
        text = str(value or "").strip()
        if "(" in text:
            text = text.split("(", 1)[0].strip()
        for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LongMemEvalItemError(f"invalid date: {value!r}") from exc
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
