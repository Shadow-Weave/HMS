"""Semantic persistence interfaces consumed by Retain planning."""

from __future__ import annotations

from typing import Protocol

from ..domain import ExistingChunkFingerprint
from .models import CommittedUnitBinding, ExistingDocument, OperationCheckpoint


class PlanningRepository(Protocol):
    """Read the minimum durable state required to build a change plan."""

    async def load_document(self, bank_id: str, document_id: str) -> ExistingDocument | None: ...

    async def load_chunks(
        self,
        bank_id: str,
        document_id: str,
    ) -> tuple[ExistingChunkFingerprint, ...]: ...

    async def load_document_unit_ids(self, bank_id: str, document_id: str) -> tuple[str, ...]: ...

    async def load_document_unit_bindings(
        self,
        bank_id: str,
        document_id: str,
        *,
        expected_unit_ids: tuple[str, ...] | None = None,
    ) -> tuple[CommittedUnitBinding, ...]: ...


class CheckpointStore(Protocol):
    """Own async-operation metadata outside the core memory UoW."""

    async def recover_document_ids(self, operation_id: str) -> tuple[str, ...]: ...

    async def recover(self, operation_id: str) -> OperationCheckpoint: ...

    async def record_document_id(self, operation_id: str, document_id: str) -> None: ...

    async def record_core_committed(
        self,
        operation_id: str,
        document_id: str,
        *,
        unit_ids: tuple[str, ...],
        requires_final_ann: bool,
    ) -> None: ...

    async def record_final_ann_completed(self, operation_id: str, document_id: str) -> None: ...

    async def clear_provider_batch(self, operation_id: str) -> None: ...
