"""Run a dependency-free LongMemEval-shaped RSI development smoke.

This is a development fixture, not a benchmark score.  It proves that a raw
LongMemEval-shaped item can be sanitized, replayed chronologically, promoted
as memory, and used to attest one declarative skill snapshot without exposing
its reference answer to the runtime.

Run from the repository root:

    PYTHONPATH=core/rsi-harness python core/rsi-harness/examples/longmem_rsi_smoke.py
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from dataclasses import replace

from rsi_harness import (
    DevelopmentStreamVerifier,
    HarnessAssetKind,
    HarnessCandidate,
    InMemorySkillLibrary,
    LongMemEvalAdapter,
    MemoryBackendCapabilities,
    PromotionResult,
    RunContext,
    SessionEventType,
    SessionEvolutionRunner,
    SkillEvolutionController,
    VerifierReceipt,
)
from rsi_harness.evolution import RunSeal


class MemoryBackend:
    capabilities = MemoryBackendCapabilities(search=False, promote=True, quarantine=True)

    def __init__(self) -> None:
        self.staged: list[str] = []
        self.promoted: list[str] = []
        self.quarantined: list[str] = []

    async def stage(self, candidate):
        self.staged.append(candidate.candidate_id)
        return candidate.candidate_id

    async def promote(self, candidate_ids, *, target_snapshot):
        self.promoted.extend(candidate_ids)
        return PromotionResult(target_snapshot, tuple(candidate_ids))

    async def quarantine(self, candidate_ids, reason):
        del reason
        self.quarantined.extend(candidate_ids)


class StaticSkillVerifier:
    def __init__(self, secret: bytes) -> None:
        self.secret = secret

    async def verify(self, run: RunContext, seal: RunSeal) -> VerifierReceipt:
        token = hmac.new(
            self.secret,
            f"development:{run.run_id}:{seal.candidate_digest}:0".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return VerifierReceipt(run.run_id, True, token)


def _semantic_enricher(event):
    if event.event_type is not SessionEventType.MESSAGE or event.metadata.get("role") != "user":
        return event
    return replace(
        event,
        event_type=SessionEventType.FACT_OBSERVED,
        metadata={
            **event.metadata,
            "candidate": {
                "kind": "fact",
                "namespace": "general",
                "trigger": "fact_observed",
                "content": event.observation,
            },
        },
    )


async def main() -> None:
    raw_item = {
        "question_id": "dev-q-1",
        "question": "What color do I prefer?",
        "question_date": "2025/01/03 (Fri) 10:00",
        "answer": "blue",
        "question_type": "single-session-user",
        "haystack_sessions": [
            [{"role": "user", "content": "I prefer blue.", "has_answer": True}],
            [{"role": "assistant", "content": "I will remember that preference."}],
        ],
        "haystack_dates": ["2025/01/01 (Wed) 10:00", "2025/01/02 (Thu) 10:00"],
        "haystack_session_ids": ["s-1", "s-2"],
    }
    runtime = LongMemEvalAdapter().runtime(raw_item, evaluation_id="dev-eval-1")
    backend = MemoryBackend()
    run = RunContext("dev-run-1", runtime.item_id, stream_id=runtime.query.stream_id)
    runner = SessionEvolutionRunner(run, backend, event_enricher=_semantic_enricher)
    memory_verifier = DevelopmentStreamVerifier(
        evaluator=lambda artifact: (
            len(artifact.trace.candidates) == 1
            and [session.source_position for session in artifact.trace.sessions] == [0, 1]
            and not any("answer" in str(event.metadata).lower() for event in artifact.trace.events)
        ),
        receipt_secret=b"development-secret",
    )
    await runner.prepare(runtime.sessions, query=runtime.query)
    memory_result = await runner.finalize(memory_verifier)

    skills = InMemorySkillLibrary()
    skill_run = RunContext("skill-run-1", "dev-cohort-1", harness_snapshot=skills.current_snapshot_id)
    skill_candidate = HarnessCandidate(
        candidate_id="skill-candidate-1",
        base_snapshot_id=skills.current_snapshot_id,
        kind=HarnessAssetKind.RETRIEVAL_POLICY,
        patch={"session_expansion_weight": 0.4},
        source_run_id=skill_run.run_id,
        evidence_refs=(f"development://{run.run_id}/candidate",),
    )
    secret = b"skill-secret"
    skill_verifier = StaticSkillVerifier(secret)
    skill_controller = SkillEvolutionController(
        skills,
        receipt_validator=lambda seal, receipt: hmac.compare_digest(
            receipt.receipt_token,
            hmac.new(
                secret,
                f"development:{seal.run_id}:{seal.candidate_digest}:0".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest(),
        ),
    )
    skill_result = await skill_controller.evolve(skill_run, skill_candidate, skill_verifier)

    print(
        json.dumps(
            {
                "runtime_item_id": runtime.item_id,
                "runtime_session_count": len(runtime.sessions),
                "memory_snapshot_id": memory_result.snapshot_id,
                "memory_promoted": memory_result.promotion is not None,
                "skill_snapshot_id": skill_result.current_snapshot_id,
                "skill_promoted": skill_result.promotion is not None,
                "runtime_contains_reference_answer": hasattr(runtime.query, "answer"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
