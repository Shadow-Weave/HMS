"""Contracts for the Retain ingestion pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..response_models import TokenUsage
from ..retain.types import RetainContentDict

OutboxCallback = Callable[[Any], Awaitable[None]]
CoreCommitCallback = Callable[[Any, tuple[tuple[str, ...], ...]], Awaitable[None]]


class RetainOperationInactiveError(RuntimeError):
    """A tracked Retain operation became terminal before its core write."""


@dataclass(frozen=True, slots=True)
class RetainInvocation:
    """Raw invocation received below ``MemoryEngine`` batch handling."""

    bank_id: str
    raw_contents: tuple[RetainContentDict, ...]
    request_context: Any
    batch_document_id: str | None = None
    is_first_batch: bool = True
    fact_type_override: str | None = None
    document_tags: tuple[str, ...] | None = None
    operation_id: str | None = None
    outbox_callback: OutboxCallback | None = None
    strategy: str | None = None
    sanitize_log_identifiers: bool = False


@dataclass(frozen=True, slots=True)
class RetainExecutionContext:
    """Resolved dependencies for one Retain submission shard."""

    pool: Any
    embeddings_model: Any
    llm_config: Any
    entity_resolver: Any
    format_date_fn: Callable[..., str]
    resolved_config: Any
    schema: str | None = None
    db_semaphore: asyncio.Semaphore | None = None


@dataclass(frozen=True, slots=True)
class RetainOutcome:
    """Result returned by the Retain pipeline."""

    unit_ids_by_input: list[list[str]]
    usage: TokenUsage
    processed_content_tokens: int | None

    def as_tuple(self) -> tuple[list[list[str]], TokenUsage, int | None]:
        """Return the tuple consumed by ``MemoryEngine``."""

        return self.unit_ids_by_input, self.usage, self.processed_content_tokens


class RetainPipeline(Protocol):
    """Interface implemented by the Retain service."""

    async def retain(
        self,
        invocation: RetainInvocation,
        execution: RetainExecutionContext,
    ) -> RetainOutcome: ...
