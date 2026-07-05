"""
Test module for causal link strategies.

Run with: pytest tests/test_causal_strategies.py -v
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from hms_api.engine.search.causal_strategies import (
    CausalContext,
    CausalLinkStrategy,
    CausalNeighbor,
    CausalScore,
    causal_registry,
)


class TestCausalNeighbors:
    """Test CausalNeighbor dataclass."""

    def test_basic_neighbor(self):
        """Test creating a basic causal neighbor."""
        neighbor = CausalNeighbor(
            neighbor_id="test-id-123",
            link_weight=0.8,
            link_type="caused_by",
        )

        assert neighbor.neighbor_id == "test-id-123"
        assert neighbor.link_weight == 0.8
        assert neighbor.link_type == "caused_by"
        assert neighbor.provenance is None

    def test_neighbor_with_provenance(self):
        """Test creating a neighbor with provenance data."""
        neighbor = CausalNeighbor(
            neighbor_id="test-id-456",
            link_weight=0.5,
            link_type="causes",
            provenance={
                "text": "This is a test fact",
                "event_date": datetime(2024, 1, 15),
            },
        )

        assert neighbor.provenance is not None
        assert neighbor.provenance["text"] == "This is a test fact"
        assert neighbor.provenance["event_date"] == datetime(2024, 1, 15)


class TestCausalContext:
    """Test CausalContext dataclass."""

    def test_basic_context(self):
        """Test creating a basic causal context."""
        context = CausalContext(
            parent_score=0.8,
            distance=2,
        )

        assert context.parent_score == 0.8
        assert context.distance == 2
        assert context.parent_date is None
        assert context.current_date is None

    def test_context_with_dates(self):
        """Test context with temporal information."""
        parent_date = datetime(2024, 1, 10)
        current_date = datetime(2024, 1, 12)

        context = CausalContext(
            parent_score=0.9,
            parent_date=parent_date,
            current_date=current_date,
            distance=1,
        )

        assert context.parent_date == parent_date
        assert context.current_date == current_date


class TestCausalScore:
    """Test CausalScore computation."""

    def test_basic_score(self):
        """Test basic causal score."""
        score = CausalScore(
            base_score=0.64,
            components={
                "parent_score": 0.8,
                "link_weight": 1.0,
                "causal_boost_factor": 2.0,
                "decay_factor": 0.7,
            },
            boosted_score=0.896,
        )

        assert score.base_score == 0.64
        assert score.boosted_score == 0.896
        assert len(score.components) == 4


class TestCausalLinkStrategy:
    """Test abstract CausalLinkStrategy class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that CausalLinkStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CausalLinkStrategy()

    def test_compute_causal_score(self):
        """Test default causal score computation."""

        class TestStrategy(CausalLinkStrategy):
            @property
            def name(self):
                return "test"

            async def find_causal_neighbors(self, *args, **kwargs):
                return {}

        strategy = TestStrategy()
        context = CausalContext(parent_score=0.8, distance=1)

        score = strategy.compute_causal_score(
            parent_score=0.8,
            link_weight=1.0,
            context=context,
        )

        assert isinstance(score, CausalScore)
        assert score.base_score == 0.8
        assert score.components["parent_score"] == 0.8
        assert score.components["link_weight"] == 1.0
        assert score.boosted_score == 0.8 * 2.0 * 0.7


class TestCausalLinkRegistry:
    """Test causal link registry."""

    def test_register_strategy(self):
        """Test registering a custom strategy."""

        @causal_registry.register("test_strategy")
        class TestCausalStrategy(CausalLinkStrategy):
            @property
            def name(self):
                return "test_strategy"

            async def find_causal_neighbors(self, *args, **kwargs):
                return {}

        assert causal_registry.get("test_strategy") is not None

    def test_list_strategies(self):
        """Test listing registered strategies."""
        strategies = causal_registry.list()
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_create_strategy(self):
        """Test creating a strategy instance."""
        strategy = causal_registry.create("memory_links")
        assert strategy is not None
        assert strategy.name == "memory_links"

    def test_create_unknown_strategy_raises(self):
        """Test that creating an unknown strategy raises an error."""
        with pytest.raises(ValueError) as exc_info:
            causal_registry.create("nonexistent_strategy")

        assert "Unknown causal strategy" in str(exc_info.value)


class TestMemoryLinksCausalStrategy:
    """Test MemoryLinksCausalStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Create a MemoryLinksCausalStrategy instance."""
        from hms_api.engine.search.causal_implementations import MemoryLinksCausalStrategy

        return MemoryLinksCausalStrategy(
            causal_boost_factor=2.0,
            decay_factor=0.7,
        )

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "memory_links"
        assert strategy._causal_boost_factor == 2.0
        assert strategy._decay_factor == 0.7

    def test_custom_boost_factor(self, strategy):
        """Test custom boost factor."""
        context = CausalContext(parent_score=0.5, distance=1)
        score = strategy.compute_causal_score(
            parent_score=0.5,
            link_weight=1.0,
            context=context,
        )

        assert score.components["causal_boost_factor"] == 2.0
        assert score.boosted_score == 0.5 * 2.0 * 0.7

    @pytest.mark.asyncio
    async def test_find_causal_neighbors_empty_input(self, strategy):
        """Test with empty seed_ids."""
        conn = AsyncMock()
        neighbors = await strategy.find_causal_neighbors(
            conn=conn,
            seed_ids=[],
            bank_id="test-bank",
            fact_type="event",
        )

        assert neighbors == {}


class TestTemporalCausalStrategy:
    """Test TemporalCausalStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Create a TemporalCausalStrategy instance."""
        from hms_api.engine.search.causal_implementations import TemporalCausalStrategy

        return TemporalCausalStrategy(
            time_window_hours=24,
            temporal_boost_factor=1.5,
            decay_factor=0.7,
        )

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "temporal_causal"
        assert strategy._time_window_hours == 24
        assert strategy._temporal_boost_factor == 1.5

    def test_temporal_proximity_calculation(self, strategy):
        """Test temporal proximity calculation."""
        parent_date = datetime(2024, 1, 15, 10, 0)
        current_date = datetime(2024, 1, 15, 18, 0)

        proximity = strategy._calculate_temporal_proximity(parent_date, current_date)

        assert 0.0 < proximity < 1.0
        assert proximity > 0.5

    def test_outside_time_window(self, strategy):
        """Test proximity when outside time window."""
        parent_date = datetime(2024, 1, 15, 0, 0)
        current_date = datetime(2024, 1, 17, 0, 0)

        proximity = strategy._calculate_temporal_proximity(parent_date, current_date)

        assert proximity == 0.0


class TestCausalScoreWithContext:
    """Test causal score computation with various contexts."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy with default settings."""

        class TestStrategy(CausalLinkStrategy):
            @property
            def name(self):
                return "test"

            async def find_causal_neighbors(self, *args, **kwargs):
                return {}

        return TestStrategy()

    def test_score_without_context(self, strategy):
        """Test score computation without context."""
        score = strategy.compute_causal_score(
            parent_score=1.0,
            link_weight=1.0,
        )

        assert score.boosted_score == 1.0 * 1.0 * 1.0

    def test_score_with_context_distance_zero(self, strategy):
        """Test score with context having distance=0 (no decay)."""
        context = CausalContext(parent_score=1.0, distance=0)
        score = strategy.compute_causal_score(
            parent_score=1.0,
            link_weight=1.0,
            context=context,
        )

        assert score.boosted_score == 1.0 * 2.0 * 1.0

    def test_score_with_context_distance_nonzero(self, strategy):
        """Test score with context having distance>0 (applies decay)."""
        context = CausalContext(parent_score=1.0, distance=1)
        score = strategy.compute_causal_score(
            parent_score=1.0,
            link_weight=1.0,
            context=context,
        )

        assert score.boosted_score == 1.0 * 2.0 * 0.7


class TestModuleIntegration:
    """Test integration with module manager."""

    def test_config_with_causal_strategy(self):
        """Test SearchConfig with causal strategy."""
        from hms_api.engine.search import SearchConfig

        config = SearchConfig(
            causal_strategy="memory_links",
            causal_params={
                "memory_links": {
                    "causal_boost_factor": 3.0,
                    "decay_factor": 0.6,
                }
            },
        )

        assert config.causal_strategy == "memory_links"
        assert "memory_links" in config.causal_params

    def test_manager_creates_causal_strategy(self):
        """Test that module manager can create causal strategy."""
        from hms_api.engine.search import SearchConfig, SearchModuleManager

        config = SearchConfig(causal_strategy="memory_links")
        manager = SearchModuleManager(config)

        strategy = manager.get_causal_strategy()
        assert strategy is not None
        assert strategy.name == "memory_links"

    def test_manager_list_includes_causal(self):
        """Test that list_available_strategies includes causal."""
        from hms_api.engine.search import SearchModuleManager

        manager = SearchModuleManager()
        strategies = manager.list_available_strategies()

        assert "causal" in strategies
        assert len(strategies["causal"]) > 0

    def test_override_causal_strategy(self):
        """Test overriding causal strategy."""
        from hms_api.engine.search import SearchModuleManager

        manager = SearchModuleManager()

        manager.override_strategy("causal", "memory_links")
        strategy1 = manager.get_causal_strategy()
        assert strategy1.name == "memory_links"

        manager.override_strategy("causal", "temporal_causal")
        strategy2 = manager.get_causal_strategy()
        assert strategy2.name == "temporal_causal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
