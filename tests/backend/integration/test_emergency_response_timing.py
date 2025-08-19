"""
Emergency stop response timing validation tests.

Tests SUBTASK-5.5.1.4 implementation with step [1u].
Validates that emergency stop response times meet <500ms requirement
even with SDR++ coordination overhead.

This ensures PRD-FR16 and NFR12 timing requirements are met.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.utils.safety import SafetyInterlockSystem


class TestEmergencyResponseTiming:
    """Test emergency response timing requirements."""

    @pytest.fixture
    async def safety_system(self):
        """Create safety interlock system with monitoring."""
        safety = SafetyInterlockSystem()
        await safety.start_monitoring()
        yield safety
        await safety.stop_monitoring()

    @pytest.fixture
    def dual_coordinator(self):
        """Create dual coordinator with timing simulation."""
        coordinator = DualSDRCoordinator()

        # Mock with realistic timing
        async def mock_emergency_override():
            await asyncio.sleep(0.1)  # 100ms processing time
            return {
                "emergency_override_active": True,
                "response_time_ms": 100.0,
                "source_switched_to": "drone",
                "safety_activated": True,
            }

        coordinator.trigger_emergency_override = mock_emergency_override
        coordinator.get_health_status = AsyncMock(
            return_value={
                "coordination_latency_ms": 30.0,
                "ground_connection_status": 0.8,
                "coordination_active": True,
            }
        )

        return coordinator

    @pytest.fixture
    def tcp_bridge_slow(self):
        """Create TCP bridge with slow response simulation."""
        bridge = SDRPPBridgeService()
        bridge.is_running = True

        # Simulate slow communication
        async def slow_communication_check():
            await asyncio.sleep(0.05)  # 50ms network delay
            return {"connected": True, "latency_ms": 50.0, "quality": 0.7}

        bridge.get_communication_health = slow_communication_check
        return bridge

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_without_coordination(self, safety_system):
        """Test [1u] - Baseline emergency stop timing without coordination."""
        # Measure baseline emergency stop time
        start_time = time.perf_counter()

        await safety_system.emergency_stop("Timing test baseline")

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Should be very fast without coordination overhead
        assert response_time_ms < 50.0  # Should be under 50ms baseline
        assert safety_system.emergency_stopped is True

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_with_coordination(self, safety_system, dual_coordinator):
        """Test [1u] - Emergency stop timing with coordination active."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Measure emergency stop time with coordination
        start_time = time.perf_counter()

        result = await safety_system.trigger_coordination_emergency_stop(
            "Timing test with coordination"
        )

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Must meet <500ms requirement even with coordination
        assert (
            response_time_ms < 500.0
        ), f"Emergency stop took {response_time_ms:.1f}ms, exceeds 500ms limit"

        # Should also be reasonably fast (target <200ms)
        assert (
            response_time_ms < 200.0
        ), f"Emergency stop took {response_time_ms:.1f}ms, should be under 200ms"

        # Verify emergency stop was effective
        assert result["safety_emergency_stop"] is True
        assert result["coordination_emergency_stop"]["emergency_override_active"] is True

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_under_load(self, safety_system, dual_coordinator):
        """Test [1u] - Emergency stop timing under coordination load."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Start coordination loop to simulate load
        await dual_coordinator.start()

        try:
            # Let coordination run for a bit to establish load
            await asyncio.sleep(0.1)

            # Measure emergency stop under load
            start_time = time.perf_counter()

            await safety_system.trigger_coordination_emergency_stop("Load test emergency stop")

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Must still meet timing requirement under load
            assert (
                response_time_ms < 500.0
            ), f"Emergency stop under load took {response_time_ms:.1f}ms"

        finally:
            await dual_coordinator.stop()

    @pytest.mark.asyncio
    async def test_safety_interlock_timing_with_coordination(self, safety_system, dual_coordinator):
        """Test [1u] - Individual safety interlock timing with coordination."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test timing of safety checks
        start_time = time.perf_counter()

        safety_results = await safety_system.check_all_safety()

        end_time = time.perf_counter()
        check_time_ms = (end_time - start_time) * 1000

        # Safety checks should be fast even with coordination
        assert (
            check_time_ms < 100.0
        ), f"Safety checks took {check_time_ms:.1f}ms, should be under 100ms"

        # All checks should complete
        expected_checks = [
            "mode",
            "operator",
            "signal",
            "battery",
            "geofence",
            "coordination_health",
            "dual_source_signal",
        ]
        for check_name in expected_checks:
            assert check_name in safety_results

    @pytest.mark.asyncio
    async def test_coordination_health_check_timing(self, safety_system, dual_coordinator):
        """Test [1u] - Coordination health check timing."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test coordination health check timing
        start_time = time.perf_counter()

        health_status = await safety_system.check_coordination_health()

        end_time = time.perf_counter()
        health_check_time_ms = (end_time - start_time) * 1000

        # Health check should be fast
        assert health_check_time_ms < 50.0, f"Health check took {health_check_time_ms:.1f}ms"
        assert health_status["enabled"] is True

    @pytest.mark.asyncio
    async def test_dual_source_signal_check_timing(self, safety_system, dual_coordinator):
        """Test [1u] - Dual source signal check timing."""
        # Enable coordination with signal sources
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test signal check timing
        start_time = time.perf_counter()

        signal_status = await safety_system.check_dual_source_signals()

        end_time = time.perf_counter()
        signal_check_time_ms = (end_time - start_time) * 1000

        # Signal check should be fast
        assert signal_check_time_ms < 30.0, f"Signal check took {signal_check_time_ms:.1f}ms"
        assert "safe" in signal_status

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_with_slow_network(
        self, safety_system, dual_coordinator, tcp_bridge_slow
    ):
        """Test [1u] - Emergency stop timing with slow network conditions."""
        # Setup with slow bridge
        dual_coordinator.set_dependencies(tcp_bridge=tcp_bridge_slow)
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Emergency stop should still be fast despite slow network
        start_time = time.perf_counter()

        await safety_system.trigger_coordination_emergency_stop("Slow network emergency test")

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Must meet requirement even with slow network
        assert (
            response_time_ms < 500.0
        ), f"Emergency stop with slow network took {response_time_ms:.1f}ms"

    @pytest.mark.asyncio
    async def test_repeated_emergency_stop_timing(self, safety_system, dual_coordinator):
        """Test [1u] - Repeated emergency stop timing consistency."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        response_times = []

        # Test multiple emergency stops
        for i in range(5):
            # Reset emergency stop state
            await safety_system.reset_emergency_stop()

            # Measure timing
            start_time = time.perf_counter()

            await safety_system.trigger_coordination_emergency_stop(f"Repeated test {i+1}")

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)

            # Each stop must meet timing requirement
            assert response_time_ms < 500.0

        # Check consistency - no response should be dramatically slower
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)

        # Maximum shouldn't be more than 3x average (consistency check)
        assert (
            max_time < 3 * avg_time
        ), f"Inconsistent timing: max={max_time:.1f}ms, avg={avg_time:.1f}ms"

    @pytest.mark.asyncio
    async def test_safety_timing_regression_test(self, safety_system, dual_coordinator):
        """Test [1u] - Safety timing regression test suite."""
        # This test ensures that safety timing doesn't degrade over time

        # Test without coordination (baseline)
        start_time = time.perf_counter()
        await safety_system.emergency_stop("Baseline regression test")
        baseline_time = (time.perf_counter() - start_time) * 1000

        # Reset and test with coordination
        await safety_system.reset_emergency_stop()
        safety_system.set_coordination_system(dual_coordinator, active=True)

        start_time = time.perf_counter()
        await safety_system.trigger_coordination_emergency_stop("Coordination regression test")
        coordination_time = (time.perf_counter() - start_time) * 1000

        # Coordination overhead should be reasonable (<5x baseline)
        overhead_ratio = coordination_time / baseline_time if baseline_time > 0 else float("inf")
        assert (
            overhead_ratio < 5.0
        ), f"Coordination overhead too high: {overhead_ratio:.1f}x baseline"

        # Both must meet absolute requirement
        assert baseline_time < 500.0
        assert coordination_time < 500.0
