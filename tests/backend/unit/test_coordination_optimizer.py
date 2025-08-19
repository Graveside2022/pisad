"""
Test suite for coordination optimization utilities.

Tests coordination latency measurement, statistical analysis, and performance monitoring
with authentic system integration validation.

User Story: Epic 5 Story 5.3 TASK-5.3.3 - Performance Optimization
PRD Reference: PRD-NFR2 (<100ms latency), PRD-NFR12 (deterministic timing)
"""

import asyncio

import pytest

from src.backend.utils.coordination_optimizer import (
    CoordinationLatencyTracker,
)


class TestCoordinationLatencyTracker:
    """Test coordination latency measurement and monitoring."""

    @pytest.fixture
    def latency_tracker(self):
        """Create latency tracker for testing."""
        return CoordinationLatencyTracker(
            max_samples=100, alert_threshold_ms=50.0, warning_threshold_ms=30.0
        )

    def test_latency_tracker_initialization(self, latency_tracker):
        """Test latency tracker initializes with correct configuration."""
        assert latency_tracker.max_samples == 100
        assert latency_tracker.alert_threshold_ms == 50.0
        assert latency_tracker.warning_threshold_ms == 30.0
        assert len(latency_tracker.latencies) == 0
        assert latency_tracker.total_measurements == 0

    def test_record_latency_single_measurement(self, latency_tracker):
        """Test recording single latency measurement."""
        # RED: This should fail - latency tracker doesn't exist yet
        latency_ms = 25.5
        latency_tracker.record_latency(latency_ms)

        assert len(latency_tracker.latencies) == 1
        assert latency_tracker.latencies[0] == latency_ms
        assert latency_tracker.total_measurements == 1

    def test_record_latency_multiple_measurements(self, latency_tracker):
        """Test recording multiple latency measurements with rolling buffer."""
        # Add measurements up to max capacity
        test_latencies = [10.0, 15.0, 20.0, 25.0, 30.0]
        for latency in test_latencies:
            latency_tracker.record_latency(latency)

        assert len(latency_tracker.latencies) == 5
        assert latency_tracker.total_measurements == 5
        assert latency_tracker.latencies == test_latencies

    def test_record_latency_exceeds_buffer_size(self, latency_tracker):
        """Test latency buffer rolls over when exceeding max_samples."""
        # Set small buffer for testing
        latency_tracker.max_samples = 3

        # Add more measurements than buffer size
        test_latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        for latency in test_latencies:
            latency_tracker.record_latency(latency)

        # Should keep only last 3 measurements
        assert len(latency_tracker.latencies) == 3
        assert latency_tracker.latencies == [30.0, 40.0, 50.0]
        assert latency_tracker.total_measurements == 5

    def test_get_statistics_empty_buffer(self, latency_tracker):
        """Test statistics calculation with empty buffer."""
        stats = latency_tracker.get_statistics()

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.min_latency == 0.0
        assert stats.max_latency == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0

    def test_get_statistics_with_measurements(self, latency_tracker):
        """Test statistics calculation with actual measurements."""
        # Add test data with known statistics
        test_latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        for latency in test_latencies:
            latency_tracker.record_latency(latency)

        stats = latency_tracker.get_statistics()

        assert stats.count == 5
        assert stats.mean == 30.0  # (10+20+30+40+50)/5
        assert stats.min_latency == 10.0
        assert stats.max_latency == 50.0
        # P95 and P99 will be calculated by implementation

    def test_check_alerts_no_alert_conditions(self, latency_tracker):
        """Test alert checking with latencies below thresholds."""
        # Add latencies below warning threshold
        test_latencies = [15.0, 20.0, 25.0]
        for latency in test_latencies:
            latency_tracker.record_latency(latency)

        alerts = latency_tracker.check_alerts()
        assert len(alerts) == 0

    def test_check_alerts_warning_threshold(self, latency_tracker):
        """Test alert generation when warning threshold exceeded."""
        # Add latency above warning but below alert threshold
        latency_tracker.record_latency(35.0)  # Above 30ms warning

        alerts = latency_tracker.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == "warning"
        assert alerts[0].threshold_ms == 30.0
        assert alerts[0].measured_latency_ms == 35.0

    def test_check_alerts_critical_threshold(self, latency_tracker):
        """Test alert generation when critical threshold exceeded."""
        # Add latency above alert threshold
        latency_tracker.record_latency(55.0)  # Above 50ms alert

        alerts = latency_tracker.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == "critical"
        assert alerts[0].threshold_ms == 50.0
        assert alerts[0].measured_latency_ms == 55.0

    @pytest.mark.asyncio
    async def test_measure_async_operation_timing(self, latency_tracker):
        """Test context manager for measuring async operation timing."""

        async def test_operation():
            await asyncio.sleep(0.01)  # 10ms operation

        # Measure operation timing
        async with latency_tracker.measure() as measurement:
            await test_operation()

        # Should have recorded one measurement
        assert latency_tracker.total_measurements == 1
        assert len(latency_tracker.latencies) == 1

        # Latency should be approximately 10ms (allow some variance)
        recorded_latency = latency_tracker.latencies[0]
        assert 8.0 <= recorded_latency <= 15.0  # 10ms ± 5ms tolerance

    def test_reset_measurements(self, latency_tracker):
        """Test resetting all measurements and statistics."""
        # Add some measurements
        test_latencies = [10.0, 20.0, 30.0]
        for latency in test_latencies:
            latency_tracker.record_latency(latency)

        # Reset and verify
        latency_tracker.reset()

        assert len(latency_tracker.latencies) == 0
        assert latency_tracker.total_measurements == 0

        stats = latency_tracker.get_statistics()
        assert stats.count == 0
        assert stats.mean == 0.0
