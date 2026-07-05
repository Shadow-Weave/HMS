"""
Comprehensive test for causal link strategies module.

This script tests all functionality of the modular causal link system.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from hms_api.engine.search import (
    CausalContext,
    CausalLinkStrategy,
    CausalNeighbor,
    CausalScore,
    SearchConfig,
    SearchModuleManager,
    MemoryLinksCausalStrategy,
    TemporalCausalStrategy,
    LLMDynamicCausalStrategy,
    HybridCausalStrategy,
    causal_registry,
)


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")


def test_basic_imports():
    """Test that all imports work correctly."""
    print_section("1. Testing Basic Imports")

    assert CausalLinkStrategy is not None
    assert CausalNeighbor is not None
    assert CausalContext is not None
    assert CausalScore is not None

    print("✅ All basic types imported successfully")


def test_registry_functionality():
    """Test the causal registry."""
    print_section("2. Testing Registry Functionality")

    strategies = causal_registry.list()
    print(f"Registered strategies: {strategies}")
    assert len(strategies) >= 4, "Expected at least 4 strategies"

    assert causal_registry.get("memory_links") is not None
    assert causal_registry.get("temporal_causal") is not None
    assert causal_registry.get("llm_dynamic") is not None
    assert causal_registry.get("hybrid_causal") is not None

    print("✅ Registry functionality working correctly")


def test_causal_neighbor_creation():
    """Test creating CausalNeighbor objects."""
    print_section("3. Testing CausalNeighbor Creation")

    neighbor1 = CausalNeighbor(
        neighbor_id="test-123",
        link_weight=0.8,
        link_type="caused_by",
    )
    print(f"Basic neighbor: {neighbor1}")
    assert neighbor1.neighbor_id == "test-123"
    assert neighbor1.link_weight == 0.8

    neighbor2 = CausalNeighbor(
        neighbor_id="test-456",
        link_weight=0.5,
        link_type="causes",
        provenance={
            "text": "Sample fact",
            "event_date": datetime(2024, 1, 15),
        },
    )
    print(f"Neighbor with provenance: {neighbor2}")
    assert neighbor2.provenance is not None
    assert neighbor2.provenance["text"] == "Sample fact"

    print("✅ CausalNeighbor creation working correctly")


def test_causal_context_creation():
    """Test creating CausalContext objects."""
    print_section("4. Testing CausalContext Creation")

    context1 = CausalContext(
        parent_score=0.8,
        distance=2,
    )
    print(f"Basic context: {context1}")
    assert context1.parent_score == 0.8
    assert context1.distance == 2

    context2 = CausalContext(
        parent_score=0.9,
        parent_date=datetime(2024, 1, 10),
        current_date=datetime(2024, 1, 12),
        distance=1,
    )
    print(f"Context with dates: {context2}")
    assert context2.parent_date is not None
    assert context2.current_date is not None

    print("✅ CausalContext creation working correctly")


def test_memory_links_strategy():
    """Test MemoryLinksCausalStrategy."""
    print_section("5. Testing MemoryLinksCausalStrategy")

    strategy = MemoryLinksCausalStrategy(
        causal_boost_factor=2.0,
        decay_factor=0.7,
        max_neighbors_per_seed=20,
    )

    print(f"Strategy name: {strategy.name}")
    print(f"Boost factor: {strategy._causal_boost_factor}")
    print(f"Decay factor: {strategy._decay_factor}")
    print(f"Max neighbors: {strategy._max_neighbors_per_seed}")

    assert strategy.name == "memory_links"

    context = CausalContext(parent_score=0.8, distance=1)
    score = strategy.compute_causal_score(
        parent_score=0.8,
        link_weight=1.0,
        context=context,
    )

    print(f"\nCausal score computation:")
    print(f"  Base score: {score.base_score}")
    print(f"  Components: {score.components}")
    print(f"  Boosted score: {score.boosted_score}")

    expected_boosted = 0.8 * 1.0 * 2.0 * 0.7
    assert abs(score.boosted_score - expected_boosted) < 0.001

    print("✅ MemoryLinksCausalStrategy working correctly")


def test_temporal_causal_strategy():
    """Test TemporalCausalStrategy."""
    print_section("6. Testing TemporalCausalStrategy")

    strategy = TemporalCausalStrategy(
        time_window_hours=24,
        temporal_boost_factor=1.5,
        decay_factor=0.7,
    )

    print(f"Strategy name: {strategy.name}")
    print(f"Time window: {strategy._time_window_hours}h")
    print(f"Temporal boost: {strategy._temporal_boost_factor}")

    parent_date = datetime(2024, 1, 15, 10, 0)
    current_date = datetime(2024, 1, 15, 18, 0)
    proximity = strategy._calculate_temporal_proximity(parent_date, current_date)
    print(f"Temporal proximity (8h diff): {proximity:.3f}")
    assert 0.0 < proximity < 1.0

    far_date = datetime(2024, 1, 17, 0, 0)
    proximity_far = strategy._calculate_temporal_proximity(parent_date, far_date)
    print(f"Temporal proximity (48h diff): {proximity_far}")
    assert proximity_far == 0.0

    print("✅ TemporalCausalStrategy working correctly")


def test_custom_boost_factor():
    """Test custom boost factor configuration."""
    print_section("7. Testing Custom Boost Factor")

    strategy = MemoryLinksCausalStrategy(
        causal_boost_factor=3.0,
        decay_factor=0.5,
    )

    context = CausalContext(parent_score=1.0, distance=1)
    score = strategy.compute_causal_score(
        parent_score=1.0,
        link_weight=1.0,
        context=context,
    )

    print(f"Custom boost factor: {strategy._causal_boost_factor}")
    print(f"Custom decay factor: {strategy._decay_factor}")
    print(f"Boosted score: {score.boosted_score}")

    expected = 1.0 * 1.0 * 3.0 * 0.5
    assert abs(score.boosted_score - expected) < 0.001

    print("✅ Custom boost factor working correctly")


def test_search_config():
    """Test SearchConfig with causal strategy."""
    print_section("8. Testing SearchConfig")

    config = SearchConfig(
        retrieval_strategy="semantic_bm25",
        graph_retrieval_strategy="link_expansion",
        fusion_strategy="rrf",
        reranking_strategy="cross_encoder",
        causal_strategy="memory_links",
        causal_params={
            "memory_links": {
                "causal_boost_factor": 2.5,
                "decay_factor": 0.8,
            }
        },
    )

    print(f"Retrieval strategy: {config.retrieval_strategy}")
    print(f"Graph retrieval strategy: {config.graph_retrieval_strategy}")
    print(f"Fusion strategy: {config.fusion_strategy}")
    print(f"Reranking strategy: {config.reranking_strategy}")
    print(f"Causal strategy: {config.causal_strategy}")
    print(f"Causal params: {config.causal_params}")

    assert config.causal_strategy == "memory_links"
    assert config.causal_params["memory_links"]["causal_boost_factor"] == 2.5

    print("✅ SearchConfig working correctly")


def test_module_manager_integration():
    """Test integration with SearchModuleManager."""
    print_section("9. Testing Module Manager Integration")

    config = SearchConfig(
        causal_strategy="memory_links",
        causal_params={
            "memory_links": {
                "causal_boost_factor": 2.0,
                "decay_factor": 0.7,
            }
        },
    )

    manager = SearchModuleManager(config)

    all_strategies = manager.list_available_strategies()
    print(f"All available strategies:")
    for category, strategies in all_strategies.items():
        print(f"  {category}: {strategies}")

    assert "causal" in all_strategies
    assert "memory_links" in all_strategies["causal"]

    causal_strategy = manager.get_causal_strategy()
    print(f"\nCreated causal strategy: {causal_strategy.name}")
    assert causal_strategy.name == "memory_links"

    print("✅ Module manager integration working correctly")


def test_strategy_override():
    """Test overriding causal strategy at runtime."""
    print_section("10. Testing Strategy Override")

    manager = SearchModuleManager()

    manager.override_strategy("causal", "memory_links")
    strategy1 = manager.get_causal_strategy()
    print(f"After override to memory_links: {strategy1.name}")
    assert strategy1.name == "memory_links"

    manager.override_strategy("causal", "temporal_causal")
    strategy2 = manager.get_causal_strategy()
    print(f"After override to temporal_causal: {strategy2.name}")
    assert strategy2.name == "temporal_causal"

    print("✅ Strategy override working correctly")


def test_custom_strategy_registration():
    """Test registering a custom strategy."""
    print_section("11. Testing Custom Strategy Registration")

    @causal_registry.register("custom_test")
    class CustomCausalStrategy(CausalLinkStrategy):
        @property
        def name(self):
            return "custom_test"

        async def find_causal_neighbors(self, *args, **kwargs):
            return {}

    strategies = causal_registry.list()
    print(f"Registered strategies: {strategies}")
    assert "custom_test" in strategies

    strategy = causal_registry.create("custom_test")
    print(f"Created custom strategy: {strategy.name}")
    assert strategy.name == "custom_test"

    print("✅ Custom strategy registration working correctly")


def test_multiple_strategies_same_config():
    """Test creating multiple strategy instances with same config."""
    print_section("12. Testing Multiple Strategy Instances")

    configs = [
        {"causal_boost_factor": 1.5, "decay_factor": 0.9},
        {"causal_boost_factor": 2.5, "decay_factor": 0.5},
        {"causal_boost_factor": 3.0, "decay_factor": 0.3},
    ]

    strategies = []
    for i, cfg in enumerate(configs):
        strategy = MemoryLinksCausalStrategy(**cfg)
        strategies.append(strategy)

        context = CausalContext(parent_score=1.0, distance=1)
        score = strategy.compute_causal_score(
            parent_score=1.0,
            link_weight=1.0,
            context=context,
        )

        print(f"Strategy {i+1}: boost={cfg['causal_boost_factor']}, "
              f"decay={cfg['decay_factor']}, "
              f"score={score.boosted_score:.3f}")

    scores = [s.compute_causal_score(1.0, 1.0, CausalContext(1.0, distance=1)).boosted_score
              for s in strategies]

    print(f"  Computed scores: {[f'{s:.3f}' for s in scores]}")

    assert all(abs(s - expected) < 0.001 for s, expected in zip(scores, [1.35, 1.25, 0.90]))

    print("✅ Multiple strategy instances working correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" CAUSAL LINK STRATEGIES MODULE - COMPREHENSIVE TEST")
    print("=" * 60)

    test_basic_imports()
    test_registry_functionality()
    test_causal_neighbor_creation()
    test_causal_context_creation()
    test_memory_links_strategy()
    test_temporal_causal_strategy()
    test_custom_boost_factor()
    test_search_config()
    test_module_manager_integration()
    test_strategy_override()
    test_custom_strategy_registration()
    test_multiple_strategies_same_config()

    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED! ✅")
    print("=" * 60 + "\n")

    print("\n📋 SUMMARY:")
    print(f"  Total strategies available: {len(causal_registry.list())}")
    print(f"  Strategy names: {', '.join(causal_registry.list())}")
    print("\n🔧 Configuration Options:")
    print("  - causal_strategy: Choose from available strategies")
    print("  - causal_params: Configure boost and decay factors")
    print("\n🚀 Usage Examples:")
    print("  1. MemoryLinksCausalStrategy: Use pre-computed causal links")
    print("  2. TemporalCausalStrategy: Use temporal proximity as causal")
    print("  3. LLMDynamicCausalStrategy: Use LLM for dynamic causal reasoning")
    print("  4. HybridCausalStrategy: Combine both approaches")


if __name__ == "__main__":
    main()
