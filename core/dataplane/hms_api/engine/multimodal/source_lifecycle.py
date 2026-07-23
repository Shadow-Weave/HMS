"""Verified source-object lifecycle state for multimodal conversion.

``FileStorage.delete()`` is an at-least-once external side effect for object
stores: a timeout may arrive after the object was deleted, and a successful
response does not itself prove that a buggy/eventually-consistent backend no
longer exposes the object.  This module therefore derives provenance from a
subsequent ``exists()`` check, never from the delete response alone.

The result contains no object key or backend exception text, so it is safe to
use in bounded operation metadata and metrics.  In particular, an unavailable
``exists()`` check remains ``unknown`` rather than being reported
optimistically as retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class _LifecycleStorage(Protocol):
    async def delete(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool: ...


class SourceAvailability(str, Enum):
    """Tri-state result of an authoritative storage existence check."""

    AVAILABLE = "available"
    DELETED = "deleted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceLifecycleResult:
    availability: SourceAvailability
    delete_requested: bool
    delete_raised: bool

    @property
    def metadata_value(self) -> str:
        """Flat provenance value without pretending unknown means available."""

        if self.availability is SourceAvailability.AVAILABLE:
            return "true"
        if self.availability is SourceAvailability.DELETED:
            return "false"
        return "unknown"

    @property
    def metric_state(self) -> str:
        if self.availability is SourceAvailability.AVAILABLE:
            return "retained"
        if self.availability is SourceAvailability.DELETED:
            return "deleted"
        return "unknown"

    @property
    def reason(self) -> str | None:
        """Return a bounded code only when requested/observed state disagrees."""

        if self.availability is SourceAvailability.UNKNOWN:
            return "source.availability_unverified"
        if not self.delete_requested:
            return None if self.availability is SourceAvailability.AVAILABLE else "source.unexpectedly_missing"
        if self.availability is SourceAvailability.DELETED:
            # A response-lost delete is still complete when existence proves
            # the object is gone.
            return None
        return "source.delete_failed" if self.delete_raised else "source.delete_incomplete"


async def resolve_source_lifecycle(
    storage: _LifecycleStorage,
    key: str,
    *,
    delete_requested: bool,
) -> SourceLifecycleResult:
    """Attempt configured deletion, then determine actual availability.

    Backend exceptions are deliberately collapsed to booleans/state here;
    neither the object key nor exception message crosses this boundary.
    """

    delete_raised = False
    if delete_requested:
        try:
            await storage.delete(key)
        except Exception:
            delete_raised = True

    try:
        exists = await storage.exists(key)
    except Exception:
        availability = SourceAvailability.UNKNOWN
    else:
        availability = SourceAvailability.AVAILABLE if exists else SourceAvailability.DELETED

    return SourceLifecycleResult(
        availability=availability,
        delete_requested=delete_requested,
        delete_raised=delete_raised,
    )


__all__ = [
    "SourceAvailability",
    "SourceLifecycleResult",
    "resolve_source_lifecycle",
]
