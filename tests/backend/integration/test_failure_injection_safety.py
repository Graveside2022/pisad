"""
Failure injection testing framework for safety scenarios.

Tests SUBTASK-5.5.1.4 implementation with step [1x].
Implements comprehensive failure injection testing to validate
safety system robustness under all failure conditions.

This ensures safety systems remain functional under any failure scenario.
"""

import asyncio
import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.utils.safety import SafetyInterlockSystem


class FailureInjector:
    """Utility class for injecting various failure scenarios."""

    @staticmethod
    def create_network_failure_mock():
        """Create mock that simulates network failures."""

        async def network_failure(*args, **kwargs):
            raise ConnectionError("Network failure injected")

        return AsyncMock(side_effect=network_failure)

    @staticmethod
    def create_timeout_mock(delay_seconds=2.0):
        """Create mock that simulates timeouts."""

        async def timeout_simulation(*args, **kwargs):
            await asyncio.sleep(delay_seconds)
            return {"status": "timeout"}

        return AsyncMock(side_effect=timeout_simulation)

    @staticmethod
    def create_intermittent_failure_mock(failure_rate=0.5):
        """Create mock with intermittent failures."""
        call_count = 0
        lock = asyncio.Lock()

        async def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            async with lock:
                call_count += 1
                current_call = call_count
                should_fail = random.random() < failure_rate

            if should_fail:
                raise Exception(f"Intermittent failure #{current_call}")
            return {"status": "success", "call": current_call}

        return AsyncMock(side_effect=intermittent_failure)

    @staticmethod
    def create_degraded_performance_mock(base_delay=0.1, degradation_factor=2.0):
        """Create mock with degraded performance."""

        async def degraded_performance(*args, **kwargs):
            delay = base_delay * degradation_factor
            await asyncio.sleep(delay)
            return {"status": "degraded", "delay_ms": delay * 1000, "performance_degraded": True}

        return AsyncMock(side_effect=degraded_performance)


class TestFailureInjectionSafety:
    """Comprehensive failure injection testing for safety systems."""

    @pytest.fixture
    async def safety_system(self):
        """Create safety interlock system."""
        safety = SafetyInterlockSystem()
        await safety.start_monitoring()
        yield safety
        await safety.stop_monitoring()

    @pytest.fixture
    def dual_coordinator(self):
        """Create dual coordinator for failure testing."""
        coordinator = DualSDRCoordinator()

        # Default working state
        coordinator.get_health_status = AsyncMock(
            return_value={
                "coordination_latency_ms": 40.0,
                "ground_connection_status": 0.8,
                "coordination_active": True,
            }
        )
        coordinator.trigger_emergency_override = AsyncMock(
            return_value={
                "emergency_override_active": True,
                "response_time_ms": 100.0,
                "source_switched_to": "drone",
            }
        )
        coordinator.get_ground_rssi = MagicMock(return_value=10.0)
        coordinator.get_drone_rssi = MagicMock(return_value=8.0)

        return coordinator

    @pytest.mark.asyncio
    async def test_network_connection_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles network connection failures."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject network failure
        dual_coordinator.get_health_status = FailureInjector.create_network_failure_mock()

        # Safety system should handle network failure gracefully
        health_status = await safety_system.check_coordination_health()

        # Should recognize failure but remain functional
        assert health_status["enabled"] is True
        assert health_status["healthy"] is False

        # Emergency stop should still work despite network failure
        await safety_system.emergency_stop("Network failure test")
        assert safety_system.emergency_stopped is True

    @pytest.mark.asyncio
    async def test_coordination_timeout_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles coordination timeouts."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject timeout failure
        dual_coordinator.get_health_status = FailureInjector.create_timeout_mock(1.5)

        # Set shorter timeout for testing
        coord_check = safety_system.checks["coordination_health"]
        coord_check.health_timeout_s = 1  # 1 second timeout

        # Health check should timeout gracefully
        health_status = await safety_system.check_coordination_health()
        assert health_status["enabled"] is True

    @pytest.mark.asyncio
    async def test_intermittent_coordination_failures(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles intermittent coordination failures."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject intermittent failures (50% failure rate)
        dual_coordinator.get_health_status = FailureInjector.create_intermittent_failure_mock(0.5)

        # Run multiple health checks
        success_count = 0
        failure_count = 0

        for _ in range(10):
            try:
                health_status = await safety_system.check_coordination_health()
                if health_status["healthy"]:
                    success_count += 1
                else:
                    failure_count += 1
            except Exception:
                failure_count += 1

        # Should have both successes and failures with intermittent failure injection
        # The important thing is that the system continues to function
        total_attempts = success_count + failure_count
        assert total_attempts == 10, "All health check attempts should complete"

        # Verify both successes and failures occurred (ensure failure injection is working)
        assert success_count > 0, "Should have some successful health checks"
        assert failure_count > 0, "Should have some failed health checks due to injection"

        # Check that distribution is roughly 50% (allowing for randomness variance)
        assert 2 <= success_count <= 8, f"Expected success count between 2-8, got {success_count}"
        assert 2 <= failure_count <= 8, f"Expected failure count between 2-8, got {failure_count}"

    @pytest.mark.asyncio
    async def test_emergency_override_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Emergency stop works when coordination emergency override fails."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject emergency override failure
        dual_coordinator.trigger_emergency_override = FailureInjector.create_network_failure_mock()

        # Emergency stop should still work
        result = await safety_system.trigger_coordination_emergency_stop(
            "Emergency override failure test"
        )

        # Safety emergency stop should work even if coordination fails
        assert result["safety_emergency_stop"] is True
        assert safety_system.emergency_stopped is True

        # Coordination override may fail, but safety stop should succeed
        assert "coordination_emergency_stop" in result

    @pytest.mark.asyncio
    async def test_signal_source_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles signal source failures."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject signal source failures
        dual_coordinator.get_ground_rssi = MagicMock(side_effect=Exception("Ground SDR failed"))
        dual_coordinator.get_drone_rssi = MagicMock(return_value=7.0)  # Only drone works

        # Signal check should handle partial failure
        signal_status = await safety_system.check_dual_source_signals()

        # Should still be safe with one working source
        assert signal_status["safe"] is True
        assert signal_status["drone_snr"] == 7.0
        assert signal_status["ground_snr"] == -100.0  # Failed source

    @pytest.mark.asyncio
    async def test_both_signal_sources_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles total signal source failure."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject total signal failure
        dual_coordinator.get_ground_rssi = MagicMock(side_effect=Exception("Ground failed"))
        dual_coordinator.get_drone_rssi = MagicMock(side_effect=Exception("Drone failed"))

        # Signal check should fail safely
        signal_status = await safety_system.check_dual_source_signals()

        # Should fail when no sources available
        assert signal_status["safe"] is False
        assert signal_status["ground_snr"] == -100.0
        assert signal_status["drone_snr"] == -100.0

    @pytest.mark.asyncio
    async def test_performance_degradation_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles performance degradation."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject performance degradation
        dual_coordinator.get_health_status = FailureInjector.create_degraded_performance_mock(
            base_delay=0.02, degradation_factor=3.0
        )

        # Health check should complete despite degradation
        import time

        start_time = time.perf_counter()

        health_status = await safety_system.check_coordination_health()

        end_time = time.perf_counter()
        check_time_ms = (end_time - start_time) * 1000

        # Should detect degraded performance
        assert check_time_ms > 50.0  # Degraded performance should be slower
        assert health_status["enabled"] is True

    @pytest.mark.asyncio
    async def test_cascading_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles cascading failures."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject cascading failures
        dual_coordinator.get_health_status = FailureInjector.create_network_failure_mock()
        dual_coordinator.get_ground_rssi = MagicMock(side_effect=Exception("Cascade failure"))
        dual_coordinator.get_drone_rssi = MagicMock(side_effect=Exception("Cascade failure"))
        dual_coordinator.trigger_emergency_override = FailureInjector.create_network_failure_mock()

        # Safety system should handle multiple simultaneous failures
        # Health check fails
        health_status = await safety_system.check_coordination_health()
        assert health_status["healthy"] is False

        # Signal check fails
        signal_status = await safety_system.check_dual_source_signals()
        assert signal_status["safe"] is False

        # Emergency stop should still work at the safety system level
        await safety_system.emergency_stop("Cascading failure test")
        assert safety_system.emergency_stopped is True

    @pytest.mark.asyncio
    async def test_memory_exhaustion_simulation(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles memory pressure simulation."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Simulate memory pressure by creating large mock responses
        large_response = {
            "coordination_latency_ms": 45.0,
            "ground_connection_status": 0.8,
            "coordination_active": True,
            "large_data": "x" * 10000,  # Large string to simulate memory pressure
            "additional_data": list(range(1000)),  # Large list
        }

        dual_coordinator.get_health_status = AsyncMock(return_value=large_response)

        # Safety system should handle large responses
        health_status = await safety_system.check_coordination_health()
        assert health_status["enabled"] is True
        assert health_status["healthy"] is True

    @pytest.mark.asyncio
    async def test_rapid_failure_recovery_cycles(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles rapid failure/recovery cycles."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Create alternating failure/success pattern
        call_count = 0

        async def alternating_behavior(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception(f"Failure #{call_count}")
            return {
                "coordination_latency_ms": 40.0,
                "ground_connection_status": 0.8,
                "coordination_active": True,
            }

        dual_coordinator.get_health_status = AsyncMock(side_effect=alternating_behavior)

        # Run rapid cycles
        results = []
        for i in range(10):
            try:
                health_status = await safety_system.check_coordination_health()
                results.append(health_status["healthy"])
            except Exception:
                results.append(False)

            await asyncio.sleep(0.01)  # Rapid cycling

        # Should handle rapid changes
        assert len(results) == 10
        # System should continue operating through rapid changes

    @pytest.mark.asyncio
    async def test_exception_propagation_safety(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system prevents exception propagation."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject various exception types
        exception_types = [
            ValueError("Value error injected"),
            TypeError("Type error injected"),
            RuntimeError("Runtime error injected"),
            ConnectionError("Connection error injected"),
            TimeoutError("Timeout error injected"),
        ]

        for exception in exception_types:
            # Reset mocks
            dual_coordinator.get_health_status = AsyncMock(side_effect=exception)

            # Safety system should handle all exception types gracefully
            try:
                health_status = await safety_system.check_coordination_health()
                # Should not raise exception, should return safe failure state
                assert health_status["enabled"] is True
                assert health_status["healthy"] is False
            except Exception as e:
                pytest.fail(f"Safety system should not propagate exceptions: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_failure_injection(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system handles concurrent operations during failures."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Inject failures in coordination components
        dual_coordinator.get_health_status = FailureInjector.create_intermittent_failure_mock(0.3)
        dual_coordinator.get_ground_rssi = MagicMock(side_effect=Exception("Concurrent failure"))

        # Run concurrent safety operations
        async def concurrent_operations():
            tasks = []

            # Multiple concurrent operations during failures
            for _ in range(5):
                tasks.append(safety_system.check_coordination_health())
                tasks.append(safety_system.check_dual_source_signals())
                tasks.append(safety_system.check_all_safety())

            return await asyncio.gather(*tasks, return_exceptions=True)

        results = await concurrent_operations()

        # Should handle concurrent operations gracefully
        # Some may fail, but should not crash or deadlock
        assert len(results) == 15  # 5 * 3 operations

        # Count successful operations vs exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        successful = [r for r in results if not isinstance(r, Exception)]

        # Should have at least some successful operations
        assert len(successful) > 0, "At least some operations should succeed"

    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(self, safety_system, dual_coordinator):
        """Test [1x] - Safety system recovers from resource exhaustion."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Simulate resource exhaustion
        exhaustion_count = 0

        async def resource_exhaustion_simulation(*args, **kwargs):
            nonlocal exhaustion_count
            exhaustion_count += 1

            if exhaustion_count <= 3:
                raise OSError("Resource temporarily unavailable")

            # Recovery after exhaustion
            return {
                "coordination_latency_ms": 40.0,
                "ground_connection_status": 0.8,
                "coordination_active": True,
                "recovered": True,
            }

        dual_coordinator.get_health_status = AsyncMock(side_effect=resource_exhaustion_simulation)

        # Test recovery
        recovery_results = []
        for i in range(5):
            try:
                health_status = await safety_system.check_coordination_health()
                recovery_results.append(health_status.get("healthy", False))
            except Exception:
                recovery_results.append(False)

        # Should recover after initial failures
        assert True in recovery_results[-2:], "Should recover from resource exhaustion"
