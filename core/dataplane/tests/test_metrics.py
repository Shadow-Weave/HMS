"""Tests for metrics instrumentation."""

from unittest.mock import MagicMock, patch

import pytest

from hms_api.metrics import (
    MetricsCollector,
    MetricsCollectorBase,
    NoOpMetricsCollector,
    create_metrics_collector,
    get_metrics_collector,
    get_token_bucket,
    initialize_metrics,
)


class TestNoOpMetricsCollector:
    """Tests for the no-op metrics collector."""

    def test_record_operation_is_noop(self):
        """Test that record_operation does nothing."""
        collector = NoOpMetricsCollector()

        # Should not raise any exception
        with collector.record_operation("recall", bank_id="test_bank", source="api"):
            pass

    def test_nested_contexts_work(self):
        """Test that nested context managers work correctly."""
        collector = NoOpMetricsCollector()

        # Nested contexts should work without issues
        with collector.record_operation("reflect", bank_id="test_bank", source="api"):
            with collector.record_operation("recall", bank_id="test_bank", source="reflect"):
                pass

    def test_exception_propagates(self):
        """Test that exceptions inside context are propagated."""
        collector = NoOpMetricsCollector()

        with pytest.raises(ValueError, match="test error"):
            with collector.record_operation("recall", bank_id="test_bank"):
                raise ValueError("test error")

    def test_record_llm_call_is_noop(self):
        """Test that record_llm_call does nothing."""
        collector = NoOpMetricsCollector()

        # Should not raise any exception
        collector.record_llm_call(
            provider="openai",
            model="gpt-4",
            scope="memory",
            duration=1.5,
            input_tokens=100,
            output_tokens=50,
            success=True,
        )

    def test_multimodal_recording_is_noop(self):
        collector = NoOpMetricsCollector()

        collector.record_multimodal_pipeline(
            media_kind="video",
            stage="sample",
            duration=0.1,
            success=True,
            candidate_frames=8,
            selected_frames=4,
        )
        with collector.record_multimodal_in_flight(media_kind="video", stage="describe"):
            pass


class TestMetricsCollector:
    """Tests for the real metrics collector."""

    @pytest.fixture
    def mock_meter(self):
        """Create a mock meter for testing."""
        meter = MagicMock()
        # operation, LLM, multimodal stage, and HTTP duration histograms
        histogram_mocks = [MagicMock() for _ in range(4)]
        meter.create_histogram.side_effect = histogram_mocks
        # operation/LLM counters, eight multimodal counters, and HTTP total
        counter_mocks = [MagicMock() for _ in range(13)]
        meter.create_counter.side_effect = counter_mocks
        meter.create_up_down_counter.side_effect = [MagicMock(), MagicMock()]
        return meter

    @pytest.fixture
    def collector(self, mock_meter):
        """Create a MetricsCollector with a mock meter."""
        mock_config = MagicMock()
        mock_config.metrics_include_bank_id = False
        with (
            patch("hms_api.metrics.get_meter", return_value=mock_meter),
            patch("hms_api.config.get_config", return_value=mock_config),
        ):
            return MetricsCollector()

    def test_record_operation_records_duration(self, collector):
        """Test that record_operation records duration."""
        with collector.record_operation("recall", bank_id="test_bank", source="api"):
            pass

        # Histogram should have been called
        collector.operation_duration.record.assert_called_once()
        call_args = collector.operation_duration.record.call_args

        # First arg is duration (should be > 0)
        duration = call_args[0][0]
        assert duration >= 0

        # Second arg is attributes dict
        attributes = call_args[0][1]
        assert attributes["operation"] == "recall"
        assert "bank_id" not in attributes  # excluded by default to avoid high-cardinality OTel growth
        assert attributes["source"] == "api"
        assert attributes["success"] == "true"

    def test_record_operation_records_failure_on_exception(self, collector):
        """Test that record_operation records failure when exception occurs."""
        with pytest.raises(RuntimeError):
            with collector.record_operation("retain", bank_id="test_bank", source="api"):
                raise RuntimeError("Test error")

        # Should have recorded with success=false
        call_args = collector.operation_duration.record.call_args
        attributes = call_args[0][1]
        assert attributes["success"] == "false"

    def test_record_operation_with_budget(self, collector):
        """Test that budget is included in attributes when provided."""
        with collector.record_operation("recall", bank_id="test_bank", source="api", budget="mid"):
            pass

        call_args = collector.operation_duration.record.call_args
        attributes = call_args[0][1]
        assert attributes["budget"] == "mid"

    def test_record_operation_with_max_tokens(self, collector):
        """Test that max_tokens is included in attributes when provided."""
        with collector.record_operation("recall", bank_id="test_bank", source="api", max_tokens=4096):
            pass

        call_args = collector.operation_duration.record.call_args
        attributes = call_args[0][1]
        assert attributes["max_tokens"] == "4096"

    def test_record_operation_source_values(self, collector):
        """Test different source values: api, reflect, internal."""
        sources = ["api", "reflect", "internal"]

        for source in sources:
            collector.operation_duration.record.reset_mock()

            with collector.record_operation("recall", bank_id="test_bank", source=source):
                pass

            call_args = collector.operation_duration.record.call_args
            attributes = call_args[0][1]
            assert attributes["source"] == source

    def test_nested_contexts_track_separately(self, collector):
        """Test that nested operations are tracked separately with different sources."""
        # Simulate reflect (api) calling recall (reflect)
        with collector.record_operation("reflect", bank_id="test_bank", source="api"):
            with collector.record_operation("recall", bank_id="test_bank", source="reflect"):
                pass

        # Should have 2 calls to record
        assert collector.operation_duration.record.call_count == 2
        assert collector.operation_total.add.call_count == 2

        # Inner recall exits first, then outer reflect.
        calls = collector.operation_duration.record.call_args_list
        recall_attrs = calls[0][0][1]
        assert recall_attrs["operation"] == "recall"
        assert recall_attrs["source"] == "reflect"
        reflect_attrs = calls[1][0][1]
        assert reflect_attrs["operation"] == "reflect"
        assert reflect_attrs["source"] == "api"

    def test_multimodal_metrics_use_bounded_labels_and_attempt_counts(self, collector):
        # Multimodal labels must not consult or expose the tenant/schema.
        with patch("hms_api.metrics._get_tenant", side_effect=AssertionError("tenant label forbidden")):
            collector.record_multimodal_pipeline(
                media_kind="video",
                stage="describe",
                duration=1.25,
                success=True,
                candidate_frames=12,
                selected_frames=5,
                logical_calls=4,
                physical_attempts=6,
                input_tokens=800,
                output_tokens=200,
            )

        duration_args = collector.multimodal_stage_duration.record.call_args[0]
        assert duration_args[0] == 1.25
        assert duration_args[1] == {
            "media_kind": "video",
            "stage": "describe",
            "outcome": "succeeded",
            "reason": "none",
        }
        assert not ({"tenant", "schema", "bank_id", "document_id", "asset_hash"} & duration_args[1].keys())
        assert collector.multimodal_calls_total.add.call_count == 3
        retry_call = next(
            call
            for call in collector.multimodal_calls_total.add.call_args_list
            if call.args[1]["attempt_kind"] == "retry"
        )
        assert retry_call.args[0] == 2
        frame_counts = {
            call.args[1]["frame_kind"]: call.args[0] for call in collector.multimodal_frames_total.add.call_args_list
        }
        assert frame_counts == {"candidate": 12, "selected": 5}
        token_directions = {
            call.args[1]["direction"]: call.args[0] for call in collector.multimodal_tokens_total.add.call_args_list
        }
        assert token_directions == {"input": 800, "output": 200}

        collector.record_multimodal_pipeline(
            media_kind="image",
            stage="complete",
            duration=0.5,
            success=True,
            deduplicated=True,
            asset_outcome="accepted",
        )
        dedupe_count, dedupe_attributes = collector.multimodal_dedupe_total.add.call_args.args
        assert dedupe_count == 1
        assert dedupe_attributes == {"media_kind": "image", "cache_result": "hit"}
        accepted_count, accepted_attributes = collector.multimodal_assets_total.add.call_args.args
        assert accepted_count == 1
        assert accepted_attributes == {
            "media_kind": "image",
            "outcome": "accepted",
            "reason": "none",
        }

    def test_multimodal_failure_metrics_have_typed_bounded_reasons(self, collector):
        collector.record_multimodal_pipeline(
            media_kind="image",
            stage="describe",
            duration=0.25,
            success=False,
            reason="provider.schema_invalid",
            logical_calls=1,
            physical_attempts=2,
            asset_outcome="rejected",
        )

        _, rejected = collector.multimodal_assets_total.add.call_args.args
        assert rejected == {
            "media_kind": "image",
            "outcome": "rejected",
            "reason": "schema_invalid",
        }
        collector.multimodal_schema_failures_total.add.assert_called_once_with(
            1,
            {
                "media_kind": "image",
                "stage": "describe",
                "reason": "schema_invalid",
            },
        )
        failure_call = next(
            call
            for call in collector.multimodal_calls_total.add.call_args_list
            if call.args[1]["attempt_kind"] == "failure"
        )
        assert failure_call.args[0] == 1

        # Arbitrary request/error strings cannot become labels or leak through
        # an unknown future call site.
        sentinel = "data:image/png;base64,PRIVATE_TENANT_PAYLOAD"
        collector.record_multimodal_pipeline(
            media_kind=sentinel,
            stage=sentinel,
            duration=0.0,
            success=False,
            reason=sentinel,
            asset_outcome="rejected",
        )
        all_attributes = [
            call.args[1]
            for instrument in (
                collector.multimodal_stage_duration,
                collector.multimodal_assets_total,
                collector.multimodal_schema_failures_total,
            )
            for call in instrument.record.call_args_list + instrument.add.call_args_list
        ]
        assert all(sentinel not in str(attributes) for attributes in all_attributes)
        assert collector.multimodal_stage_duration.record.call_args.args[1] == {
            "media_kind": "other",
            "stage": "other",
            "outcome": "failed",
            "reason": "other",
        }

    def test_multimodal_source_cancellation_and_in_flight_metrics(self, collector):
        collector.record_multimodal_pipeline(
            media_kind="video",
            stage="source_lifecycle",
            duration=0.02,
            success=True,
            source_state="deleted",
        )
        collector.multimodal_source_lifecycle_total.add.assert_called_once_with(
            1,
            {"media_kind": "video", "state": "deleted", "reason": "none"},
        )

        collector.record_multimodal_pipeline(
            media_kind="video",
            stage="describe",
            duration=0.4,
            success=False,
            reason="operation.cancelled",
            cancelled=True,
        )
        collector.multimodal_cancellations_total.add.assert_called_once_with(
            1,
            {"media_kind": "video", "stage": "describe", "reason": "cancelled"},
        )

        with pytest.raises(RuntimeError, match="provider stopped"):
            with collector.record_multimodal_in_flight(media_kind="video", stage="describe"):
                raise RuntimeError("provider stopped")
        assert collector.multimodal_in_flight.add.call_args_list == [
            ((1, {"media_kind": "video", "stage": "describe"}),),
            ((-1, {"media_kind": "video", "stage": "describe"}),),
        ]

    def test_multimodal_stage_failure_does_not_infer_a_second_asset_outcome(self, collector):
        collector.record_multimodal_pipeline(
            media_kind="image",
            stage="complete",
            duration=0.1,
            success=True,
            asset_outcome="accepted",
        )
        collector.record_multimodal_pipeline(
            media_kind="image",
            stage="retain",
            duration=0.2,
            success=False,
            reason="retain.failed",
        )

        collector.multimodal_assets_total.add.assert_called_once_with(
            1,
            {"media_kind": "image", "outcome": "accepted", "reason": "none"},
        )

    def test_record_operation_includes_bank_id_when_enabled(self):
        """Test that bank_id is included in attributes when metrics_include_bank_id is enabled."""
        mock_config = MagicMock()
        mock_config.metrics_include_bank_id = True
        with (
            patch("hms_api.metrics.get_meter") as mock_get_meter,
            patch("hms_api.config.get_config", return_value=mock_config),
        ):
            mock_get_meter.return_value = MagicMock()
            collector = MetricsCollector()

        with collector.record_operation("recall", bank_id="test_bank", source="api"):
            pass

        attributes = collector.operation_duration.record.call_args[0][1]
        assert attributes["bank_id"] == "test_bank"


class TestGetMetricsCollector:
    """Tests for the get_metrics_collector function."""

    def test_returns_noop_by_default(self):
        """Test that get_metrics_collector returns NoOpMetricsCollector by default."""
        # Reset global state
        import hms_api.metrics as metrics_module

        original_collector = metrics_module._metrics_collector

        try:
            metrics_module._metrics_collector = NoOpMetricsCollector()
            collector = get_metrics_collector()
            assert isinstance(collector, NoOpMetricsCollector)
        finally:
            metrics_module._metrics_collector = original_collector


class TestMetricsCollectorBase:
    """Tests for the MetricsCollectorBase abstract class."""

    def test_is_abstract(self):
        """Test that MetricsCollectorBase methods are abstract."""

        # Create a class that inherits but doesn't implement
        class IncompleteCollector(MetricsCollectorBase):
            pass

        collector = IncompleteCollector()

        # Abstract methods should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            with collector.record_operation("test", "test"):
                pass

        with pytest.raises(NotImplementedError):
            collector.record_llm_call("test", "test", "test", 1.0)


class TestGetTokenBucket:
    """Tests for the get_token_bucket function."""

    def test_bucket_0_100(self):
        """Test tokens < 100 return '0-100' bucket."""
        assert get_token_bucket(0) == "0-100"
        assert get_token_bucket(50) == "0-100"
        assert get_token_bucket(99) == "0-100"

    def test_bucket_100_500(self):
        """Test tokens 100-499 return '100-500' bucket."""
        assert get_token_bucket(100) == "100-500"
        assert get_token_bucket(250) == "100-500"
        assert get_token_bucket(499) == "100-500"

    def test_bucket_500_1k(self):
        """Test tokens 500-999 return '500-1k' bucket."""
        assert get_token_bucket(500) == "500-1k"
        assert get_token_bucket(750) == "500-1k"
        assert get_token_bucket(999) == "500-1k"

    def test_bucket_1k_5k(self):
        """Test tokens 1000-4999 return '1k-5k' bucket."""
        assert get_token_bucket(1000) == "1k-5k"
        assert get_token_bucket(2500) == "1k-5k"
        assert get_token_bucket(4999) == "1k-5k"

    def test_bucket_5k_10k(self):
        """Test tokens 5000-9999 return '5k-10k' bucket."""
        assert get_token_bucket(5000) == "5k-10k"
        assert get_token_bucket(7500) == "5k-10k"
        assert get_token_bucket(9999) == "5k-10k"

    def test_bucket_10k_50k(self):
        """Test tokens 10000-49999 return '10k-50k' bucket."""
        assert get_token_bucket(10000) == "10k-50k"
        assert get_token_bucket(25000) == "10k-50k"
        assert get_token_bucket(49999) == "10k-50k"

    def test_bucket_50k_plus(self):
        """Test tokens >= 50000 return '50k+' bucket."""
        assert get_token_bucket(50000) == "50k+"
        assert get_token_bucket(100000) == "50k+"
        assert get_token_bucket(1000000) == "50k+"


class TestLLMMetrics:
    """Tests for LLM-specific metrics recording."""

    @pytest.fixture
    def mock_meter(self):
        """Create a mock meter for testing."""
        meter = MagicMock()
        # operation, LLM, multimodal stage, and HTTP duration histograms
        histogram_mocks = [MagicMock() for _ in range(4)]
        meter.create_histogram.side_effect = histogram_mocks
        # operation/LLM counters, eight multimodal counters, and HTTP total
        counter_mocks = [MagicMock() for _ in range(13)]
        meter.create_counter.side_effect = counter_mocks
        meter.create_up_down_counter.side_effect = [MagicMock(), MagicMock()]
        return meter

    @pytest.fixture
    def collector(self, mock_meter):
        """Create a MetricsCollector with a mock meter."""
        mock_config = MagicMock()
        mock_config.metrics_include_bank_id = False
        with (
            patch("hms_api.metrics.get_meter", return_value=mock_meter),
            patch("hms_api.config.get_config", return_value=mock_config),
        ):
            return MetricsCollector()

    def test_record_llm_call_records_duration(self, collector):
        """Test that record_llm_call records duration."""
        collector.record_llm_call(
            provider="openai",
            model="gpt-4",
            scope="memory",
            duration=1.5,
            input_tokens=100,
            output_tokens=50,
            success=True,
        )

        # LLM duration histogram should be called
        collector.llm_duration.record.assert_called_once()
        call_args = collector.llm_duration.record.call_args

        # First arg is duration
        assert call_args[0][0] == 1.5

        # Second arg is attributes dict
        attributes = call_args[0][1]
        assert attributes["provider"] == "openai"
        assert attributes["model"] == "gpt-4"
        assert attributes["scope"] == "memory"
        assert attributes["success"] == "true"

    def test_record_llm_call_records_failure(self, collector):
        """Test that record_llm_call records failure status."""
        collector.record_llm_call(
            provider="anthropic",
            model="claude-3",
            scope="reflect",
            duration=0.5,
            success=False,
        )

        # Check success is false
        call_args = collector.llm_duration.record.call_args
        attributes = call_args[0][1]
        assert attributes["success"] == "false"

    def test_record_llm_call_records_tokens_with_buckets(self, collector):
        """Test that record_llm_call records tokens with bucket labels."""
        collector.record_llm_call(
            provider="openai",
            model="gpt-4",
            scope="memory",
            duration=1.0,
            input_tokens=2500,  # Should be "1k-5k" bucket
            output_tokens=150,  # Should be "100-500" bucket
            success=True,
        )

        # Input tokens should be recorded with bucket
        collector.llm_tokens_input.add.assert_called_once()
        input_call = collector.llm_tokens_input.add.call_args
        assert input_call[0][0] == 2500
        assert input_call[0][1]["token_bucket"] == "1k-5k"

        # Output tokens should be recorded with bucket
        collector.llm_tokens_output.add.assert_called_once()
        output_call = collector.llm_tokens_output.add.call_args
        assert output_call[0][0] == 150
        assert output_call[0][1]["token_bucket"] == "100-500"

    def test_record_llm_call_skips_zero_tokens(self, collector):
        """Test that zero token values don't record."""
        collector.record_llm_call(
            provider="openai",
            model="gpt-4",
            scope="memory",
            duration=1.0,
            input_tokens=0,
            output_tokens=0,
            success=True,
        )

        # Token counters should not be called
        collector.llm_tokens_input.add.assert_not_called()
        collector.llm_tokens_output.add.assert_not_called()

    def test_record_llm_call_increments_call_counter(self, collector):
        """Test that record_llm_call increments the call counter."""
        collector.record_llm_call(
            provider="gemini",
            model="gemini-pro",
            scope="memory",
            duration=2.0,
            success=True,
        )

        # Call counter should be incremented
        collector.llm_calls_total.add.assert_called_once()
        call_args = collector.llm_calls_total.add.call_args
        assert call_args[0][0] == 1
        assert call_args[0][1]["provider"] == "gemini"
        assert call_args[0][1]["model"] == "gemini-pro"
        assert call_args[0][1]["scope"] == "memory"

    def test_record_llm_call_different_scopes(self, collector):
        """Test recording LLM calls with different scopes."""
        scopes = ["memory", "reflect", "consolidation", "answer"]

        for scope in scopes:
            collector.llm_duration.record.reset_mock()

            collector.record_llm_call(
                provider="openai",
                model="gpt-4",
                scope=scope,
                duration=1.0,
                success=True,
            )

            call_args = collector.llm_duration.record.call_args
            attributes = call_args[0][1]
            assert attributes["scope"] == scope
