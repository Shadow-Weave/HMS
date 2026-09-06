"""Backend-neutral contracts for a live self-improvement harness.

The package intentionally contains no HMS, database, embedding, or LLM
dependency.  A memory system integrates by implementing :class:`MemoryBackend`
or by adapting its existing client to that protocol.
"""

from .backend import BackendCapabilityError, MemoryBackend, MemoryBackendCapabilities
from .benchmark import BenchmarkAccessViolation, BenchmarkFirewall, BenchmarkResult, EvaluationLease, FrozenMemoryView
from .benchmark_verifier import BenchmarkVerifierConfigurationError, BenchmarkVerifierProcess
from .candidate_factory import CandidateFactory
from .codex_adapter import CodexEventAdapter
from .datasets import LongMemEvalAdapter, LongMemEvalItemError, LongMemEvalRuntimeItem
from .development_verifier import DevelopmentVerifierProcess, VerifierConfigurationError, validate_development_receipt
from .evaluation_compare import EvaluationComparison, EvaluationComparisonError, compare_evaluation_reports
from .evaluation_snapshot import EvaluationSnapshotBuilder, EvaluationSnapshotError
from .events import EventType, SupervisorAction, SupervisorActionType, WorkerEvent
from .evolution import EvolutionSession, EvolutionStateError, RunSeal
from .frozen_benchmark import (
    FrozenBenchmarkBoundaryError,
    FrozenBenchmarkRecord,
    FrozenBenchmarkReport,
    FrozenBenchmarkRunner,
)
from .harness_evolution import (
    DefaultHarnessMutationPolicy,
    HarnessAssetKind,
    HarnessCandidate,
    HarnessEvolutionSession,
    HarnessMutationPolicy,
    HarnessPromotion,
)
from .hms_adapter import HMSMemoryAdapter
from .leakage import LeakageGuard, LeakagePolicy, LeakageViolation
from .memory_orchestrator import (
    MemoryOrchestrator,
    MemoryOrchestratorContext,
    MemoryOrchestratorPolicy,
    RuleBasedMemoryOrchestratorPolicy,
)
from .models import (
    DataClassification,
    HarnessSnapshot,
    LeakageReport,
    MemoryCandidate,
    MemoryHit,
    MemoryItem,
    MemoryNamespace,
    PromotionResult,
    RunContext,
    VerifierReceipt,
)
from .runtime import (
    EventProtocolError,
    InMemoryEventBus,
    SupervisorContext,
    SupervisorController,
    WorkerRuntime,
    serve_events,
)
from .session_evolution import (
    DevelopmentStreamArtifact,
    DevelopmentStreamVerifier,
    EventEnricher,
    SessionEvolutionResult,
    SessionEvolutionRunner,
    SessionEvolutionStateError,
    SessionEvolutionTrace,
    run_session_evolution,
)
from .session_models import (
    MemoryDecision,
    MemoryDecisionType,
    QueryInput,
    SessionEvent,
    SessionEventType,
    SessionInput,
)
from .session_runner import ChronologicalSessionRunner, SessionOrderError, replay_sessions
from .skill_library import (
    FileSkillLibrary,
    InMemorySkillLibrary,
    SkillEvolutionController,
    SkillEvolutionResult,
    SkillLibrary,
    SkillLibraryError,
)
from .snapshots import InMemorySnapshotStore, JsonSnapshotStore, SnapshotStore

__all__ = [
    "EventType",
    "DataClassification",
    "BenchmarkAccessViolation",
    "BenchmarkFirewall",
    "BenchmarkResult",
    "BenchmarkVerifierConfigurationError",
    "BenchmarkVerifierProcess",
    "BackendCapabilityError",
    "CodexEventAdapter",
    "CandidateFactory",
    "DevelopmentVerifierProcess",
    "validate_development_receipt",
    "EvaluationSnapshotBuilder",
    "EvaluationSnapshotError",
    "EvaluationComparison",
    "EvaluationComparisonError",
    "compare_evaluation_reports",
    "FrozenBenchmarkBoundaryError",
    "FrozenBenchmarkRecord",
    "FrozenBenchmarkReport",
    "FrozenBenchmarkRunner",
    "EvaluationLease",
    "EvolutionSession",
    "EvolutionStateError",
    "EventProtocolError",
    "HarnessSnapshot",
    "HMSMemoryAdapter",
    "DefaultHarnessMutationPolicy",
    "HarnessAssetKind",
    "HarnessCandidate",
    "HarnessEvolutionSession",
    "HarnessMutationPolicy",
    "HarnessPromotion",
    "InMemorySnapshotStore",
    "JsonSnapshotStore",
    "LeakageGuard",
    "LeakagePolicy",
    "LeakageReport",
    "LeakageViolation",
    "MemoryBackend",
    "MemoryBackendCapabilities",
    "MemoryDecision",
    "MemoryDecisionType",
    "MemoryCandidate",
    "MemoryHit",
    "MemoryItem",
    "MemoryNamespace",
    "PromotionResult",
    "QueryInput",
    "RunContext",
    "RunSeal",
    "SessionEvent",
    "SessionEventType",
    "SessionInput",
    "SessionOrderError",
    "ChronologicalSessionRunner",
    "replay_sessions",
    "MemoryOrchestrator",
    "MemoryOrchestratorContext",
    "MemoryOrchestratorPolicy",
    "RuleBasedMemoryOrchestratorPolicy",
    "LongMemEvalAdapter",
    "LongMemEvalItemError",
    "LongMemEvalRuntimeItem",
    "InMemoryEventBus",
    "SupervisorContext",
    "SupervisorController",
    "SnapshotStore",
    "SupervisorAction",
    "SupervisorActionType",
    "WorkerEvent",
    "WorkerRuntime",
    "DevelopmentStreamArtifact",
    "DevelopmentStreamVerifier",
    "EventEnricher",
    "SessionEvolutionResult",
    "SessionEvolutionRunner",
    "SessionEvolutionStateError",
    "SessionEvolutionTrace",
    "run_session_evolution",
    "InMemorySkillLibrary",
    "FileSkillLibrary",
    "SkillEvolutionController",
    "SkillEvolutionResult",
    "SkillLibrary",
    "SkillLibraryError",
    "FrozenMemoryView",
    "VerifierReceipt",
    "VerifierConfigurationError",
    "serve_events",
]
