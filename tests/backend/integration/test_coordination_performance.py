"""
Integration tests for coordination performance optimization.

Tests real-world performance improvements with actual services
and validates latency measurements against PRD requirements.

User Story: Epic 5 Story 5.3 TASK-5.3.3 - Performance Optimization
PRD Reference: PRD-NFR2 (<100ms latency), PRD-NFR12 (deterministic timing)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator


class TestCoordinationPerformanceIntegration:
    """Integration tests for coordination performance optimization."""

    @pytest.fixture
    async def coordinator_with_tracking(self):
        """Create coordinator with integrated latency tracking."""
        coordinator = DualSDRCoordinator()

        # Mock dependencies for testing
        signal_processor = MagicMock()
        signal_processor.get_current_rssi = MagicMock(return_value=-60.0)

        tcp_bridge = MagicMock()
        tcp_bridge.is_running = True
        tcp_bridge.get_ground_rssi = MagicMock(return_value=-55.0)

        safety_manager = MagicMock()

        coordinator.set_dependencies(
            signal_processor=signal_processor, tcp_bridge=tcp_bridge, safety_manager=safety_manager
        )

        return coordinator

    @pytest.mark.asyncio
    async def test_coordination_latency_measurement_integration(self, coordinator_with_tracking):
        """Test latency measurement integration with actual coordination."""
        # Start coordinator
        await coordinator_with_tracking.start()

        # Let it run for a short period to collect measurements
        await asyncio.sleep(0.2)  # Run for 200ms to get multiple cycles

        # Stop coordinator
        await coordinator_with_tracking.stop()

        # Verify latency measurements were collected
        assert hasattr(coordinator_with_tracking, "_latency_tracker")
        tracker = coordinator_with_tracking._latency_tracker
        assert tracker.total_measurements > 0

        # Verify latencies are within PRD requirements
        stats = tracker.get_statistics()
        assert stats.mean <= 50.0  # Target <50ms coordination latency
        assert stats.p95 <= 100.0  # PRD requirement <100ms

        # Check no critical alerts
        alerts = tracker.check_alerts()
        critical_alerts = [a for a in alerts if a.level == "critical"]
        assert len(critical_alerts) == 0

    @pytest.mark.asyncio
    async def test_coordination_decision_timing(self, coordinator_with_tracking):
        """Test individual coordination decision timing."""
        coordinator = coordinator_with_tracking

        # Measure coordination decision timing
        start_time = time.perf_counter()
        await coordinator.make_coordination_decision()
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Verify decision latency meets requirements
        assert latency_ms <= 50.0  # Target coordination latency
        assert latency_ms >= 0.1  # Sanity check - should take some time

    @pytest.mark.asyncio
    async def test_performance_under_load(self, coordinator_with_tracking):
        """Test coordination performance under sustained load."""
        coordinator = coordinator_with_tracking

        # Simulate multiple rapid coordination decisions
        latencies = []
        for _ in range(50):  # 50 decisions
            start_time = time.perf_counter()
            await coordinator.make_coordination_decision()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

            # Small delay to simulate real operation
            await asyncio.sleep(0.01)

        # Verify all decisions met latency requirements
        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)

        assert max_latency <= 100.0  # PRD requirement
        assert avg_latency <= 50.0  # Target average

        # Verify latency consistency (low variance)
        variance = sum((latency - avg_latency) ** 2 for latency in latencies) / len(latencies)
        std_dev = variance**0.5
        assert std_dev <= 10.0  # Low variance for deterministic timing

    @pytest.mark.asyncio
    async def test_fallback_performance(self, coordinator_with_tracking):
        """Test performance during fallback scenarios."""
        coordinator = coordinator_with_tracking

        # Simulate communication loss scenario
        coordinator._tcp_bridge.is_running = False

        # Measure fallback decision timing
        start_time = time.perf_counter()
        await coordinator.make_coordination_decision()
        end_time = time.perf_counter()

        fallback_latency_ms = (end_time - start_time) * 1000

        # Verify fallback still meets timing requirements
        assert fallback_latency_ms <= 50.0  # Should be even faster (no network)
        assert coordinator.active_source == "drone"  # Should fall back to drone
        assert coordinator.fallback_active

    @pytest.mark.asyncio
    async def test_priority_manager_performance(self, coordinator_with_tracking):
        """Test priority manager integration performance."""
        coordinator = coordinator_with_tracking

        # Ensure priority manager is available
        assert coordinator._priority_manager is not None

        # Mock priority manager decision
        mock_decision = MagicMock()
        mock_decision.switch_recommended = True
        mock_decision.selected_source = "ground"
        mock_decision.reason = "Ground signal stronger"
        mock_decision.latency_ms = 2.5

        coordinator._priority_manager.evaluate_source_switch = AsyncMock(return_value=mock_decision)

        # Measure coordination with priority manager
        start_time = time.perf_counter()
        await coordinator.make_coordination_decision()
        end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000

        # Verify total latency includes priority manager overhead but stays within limits
        assert total_latency_ms <= 50.0  # Total coordination latency
        assert coordinator.active_source == "ground"  # Decision should be applied
