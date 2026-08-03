"""Identifier redaction helpers for trusted Retain ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REDACTED_IDENTIFIER = "<redacted>"


@dataclass(frozen=True, slots=True)
class IdentifierSanitizer:
    """Redact request identifiers from logs and persisted error messages."""

    enabled: bool = False
    identifiers: tuple[str, ...] = ()

    @classmethod
    def from_values(
        cls,
        *,
        enabled: bool,
        values: tuple[Any, ...] = (),
    ) -> IdentifierSanitizer:
        identifiers = tuple(value for item in values if item is not None for value in (str(item),) if value)
        return cls(enabled=enabled, identifiers=identifiers)

    def identifier(self, value: Any) -> str:
        """Return one identifier in a log-safe form."""

        if value is None:
            return "None"
        text = str(value)
        if self.enabled and text:
            return REDACTED_IDENTIFIER
        return text

    def text(self, value: Any, *, extra_identifiers: tuple[Any, ...] = ()) -> str:
        """Redact every known identifier from arbitrary diagnostic text."""

        message = str(value)
        if not self.enabled:
            return message
        extras = tuple(text for item in extra_identifiers if item is not None for text in (str(item),) if text)
        for identifier in sorted((*self.identifiers, *extras), key=len, reverse=True):
            message = message.replace(identifier, REDACTED_IDENTIFIER)
        return message


__all__ = ["IdentifierSanitizer", "REDACTED_IDENTIFIER"]
