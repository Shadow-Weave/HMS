"""Show a no-leakage H0 -> H1 improvement on a tiny LongMemEval fixture.

The reference answer is used only by the final scoring function.  It never
enters ``LongMemEvalRuntimeItem``, the worker events, the candidate, or the
development verifier artifact.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from rsi_harness import (
    DevelopmentStreamVerifier,
    LongMemEvalAdapter,
    MemoryBackendCapabilities,
    MemoryHit,
    MemoryNamespace,
    PromotionResult,
    RunContext,
    SessionEventType,
    SessionEvolutionRunner,
)


class SearchableDevelopmentBackend:
    capabilities = MemoryBackendCapabilities(search=True, promote=True, quarantine=True)

    def __init__(self) -> None:
        self.staged = {}
        self.promoted = {}
        self.quarantined = []

    async def stage(self, candidate):
        self.staged[candidate.candidate_id] = candidate
        return candidate.candidate_id

    async def promote(self, candidate_ids, *, target_snapshot):
        for candidate_id in candidate_ids:
            self.promoted[candidate_id] = self.staged[candidate_id]
        return PromotionResult(target_snapshot, tuple(candidate_ids))

    async def quarantine(self, candidate_ids, reason):
        del reason
        self.quarantined.extend(candidate_ids)

    async def search(self, namespace, query, *, token_budget, filters=None):
        del token_budget, filters
        if namespace not in (MemoryNamespace.GENERAL, MemoryNamespace.PROJECT):
            return ()
        terms = set(query.lower().split())
        return tuple(
            MemoryHit(candidate.candidate_id, candidate.content, 1.0, candidate.namespace)
            for candidate in self.promoted.values()
            if terms.intersection(candidate.content.lower().split())
        )


def _enrich(event):
    if event.event_type is not SessionEventType.MESSAGE or event.metadata.get("role") != "user":
        return event
    return replace(
        event,
        event_type=SessionEventType.FACT_OBSERVED,
        metadata={
            **event.metadata,
            "candidate": {
                "kind": "preference",
                "namespace": "general",
                "trigger": "fact_observed",
                "content": event.observation,
            },
        },
    )


async def main() -> None:
    raw_item = {
        "question_id": "h0-h1-q1",
        "question": "What color do I prefer?",
        "question_date": "2025/01/03 (Fri) 10:00",
        "answer": "I prefer blue.",
        "question_type": "single-session-user",
        "haystack_sessions": [[{"role": "user", "content": "I prefer blue."}]],
        "haystack_dates": ["2025/01/01 (Wed) 10:00"],
        "haystack_session_ids": ["h0-h1-s1"],
    }
    runtime = LongMemEvalAdapter().runtime(raw_item, evaluation_id="h0-h1-dev")
    backend = SearchableDevelopmentBackend()
    run = RunContext("h0-h1-run", runtime.item_id, stream_id=runtime.query.stream_id)
    runner = SessionEvolutionRunner(run, backend, event_enricher=_enrich)
    verifier = DevelopmentStreamVerifier(
        evaluator=lambda artifact: len(artifact.trace.candidates) == 1,
        receipt_secret=b"h0-h1-development-secret",
    )
    await runner.prepare(runtime.sessions, query=runtime.query)

    # H0 is measured before the candidate is promoted.  The gold answer is
    # intentionally outside the runtime and verifier boundary.
    h0_answer = "unknown"
    h0_passed = h0_answer == raw_item["answer"]
    result = await runner.finalize(verifier)
    h1_hits = await backend.search(MemoryNamespace.GENERAL, runtime.query.query, token_budget=32)
    h1_answer = h1_hits[0].text if h1_hits else "unknown"
    h1_passed = h1_answer == raw_item["answer"]
    print(
        json.dumps(
            {
                "h0_passed": h0_passed,
                "h1_passed": h1_passed,
                "delta": int(h1_passed) - int(h0_passed),
                "memory_promoted": result.promotion is not None,
                "runtime_contains_reference_answer": hasattr(runtime.query, "answer")
                or any("answer" in str(message).lower() for session in runtime.sessions for message in session.messages),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
