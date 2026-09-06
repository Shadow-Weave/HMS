from __future__ import annotations

from rsi_harness import LeakageGuard, LeakagePolicy, MemoryCandidate, MemoryNamespace


def make_candidate(content: str, metadata=None):
    return MemoryCandidate(
        candidate_id="candidate-1",
        kind="procedure",
        namespace=MemoryNamespace.GENERAL,
        trigger="error pattern",
        content=content,
        source_run_id="run-1",
        evidence_refs=("trace://run-1/1",),
        metadata=metadata or {},
    )


def test_secret_and_verifier_metadata_are_rejected():
    guard = LeakageGuard(LeakagePolicy(forbidden_literals=("hidden answer",)))
    report = guard.inspect(make_candidate("hidden answer", {"verifier_output": "pass"}))
    assert not report.accepted
    assert report.classification.value == "benchmark_secret"


def test_private_key_is_rejected_as_sensitive():
    report = LeakageGuard().inspect(make_candidate("-----BEGIN PRIVATE KEY-----"))
    assert not report.accepted
    assert report.classification.value == "sensitive"


def test_accepted_general_candidate_has_no_findings():
    report = LeakageGuard().inspect(make_candidate("When X happens, retry after checking Y."))
    assert report.accepted
    assert report.findings == ()
