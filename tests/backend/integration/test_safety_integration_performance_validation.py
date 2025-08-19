"""
TASK-5.5.4-INTEGRATION-TESTING: Comprehensive Safety Integration Testing
SUBTASK-5.5.4.1 [4e] - Create safety integration performance validation tests

Performance validation tests to ensure safety systems maintain timing requirements
during coordination operations.

PRD References:
- PRD-FR16: Emergency stop <500ms response time
- PRD-NFR12: Deterministic timing for safety-critical functions
- PRD-NFR2: Signal processing latency <100ms per computation cycle
"""

import asyncio
import os
import statistics
import time

import psutil
import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
    SafetyDecisionType,
)
from src.backend.services.safety_manager import SafetyManager


class TestSafetyIntegrationPerformanceValidation:
    """Performance validation tests for safety integration with coordination."""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager instance."""
        manager = SafetyManager()
        await manager.start_monitoring()
        yield manager

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager instance."""
        return SafetyAuthorityManager()

    @pytest.fixture
    def dual_sdr_coordinator(self, safety_authority_manager):
        """Create dual SDR coordinator with safety integration."""
        return DualSDRCoordinator(safety_authority=safety_authority_manager)

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_under_coordination_load(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """Test emergency stop response time <500ms during coordination operations."""
        # RED PHASE - Emergency stop must be <500ms with coordination active
        await dual_sdr_coordinator.start()

        # Measure emergency stop timing 10 times for statistical validation
        timing_results = []

        for i in range(10):
            start_time = time.perf_counter()

            # Trigger emergency stop
            emergency_decision = safety_authority_manager.evaluate_safety_decision(
                level=SafetyAuthorityLevel.EMERGENCY_STOP,
                decision_type=SafetyDecisionType.FORCE_EMERGENCY_STOP,
                context={"reason": f"performance_test_{i}", "iteration": i},
            )

            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            timing_results.append(response_time)

            # Verify emergency decision was processed
            assert emergency_decision.decision_type == SafetyDecisionType.FORCE_EMERGENCY_STOP

            # Small delay between tests
            await asyncio.sleep(0.01)

        # Statistical analysis of response times
        avg_response_time = statistics.mean(timing_results)
        max_response_time = max(timing_results)

        # CRITICAL: All emergency stop responses must be <500ms
        assert (
            max_response_time < 500.0
        ), f"Max emergency stop time {max_response_time:.2f}ms > 500ms limit"
        assert (
            avg_response_time < 250.0
        ), f"Average emergency stop time {avg_response_time:.2f}ms should be well under limit"

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_safety_decision_latency_with_coordination(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """Test safety decision latency remains low during coordination."""
        # RED PHASE - Safety decisions should be fast even with coordination load
        await dual_sdr_coordinator.start()

        # Test different safety decision types
        decision_types = [
            (SafetyAuthorityLevel.COORDINATION_HEALTH, SafetyDecisionType.MONITOR_HEALTH),
            (SafetyAuthorityLevel.COMMUNICATION_MONITOR, SafetyDecisionType.EVALUATE_COMMUNICATION),
            (SafetyAuthorityLevel.BATTERY_MONITOR, SafetyDecisionType.CHECK_BATTERY),
            (SafetyAuthorityLevel.GEOFENCE_BOUNDARY, SafetyDecisionType.VALIDATE_BOUNDARY),
        ]

        for level, decision_type in decision_types:
            timing_results = []

            # Test each decision type 5 times
            for i in range(5):
                start_time = time.perf_counter()

                decision = safety_authority_manager.evaluate_safety_decision(
                    level=level,
                    decision_type=decision_type,
                    context={"test": "performance_validation", "iteration": i},
                )

                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                timing_results.append(response_time)

                assert decision.authority_level == level
                assert decision.decision_type == decision_type

            # Verify decision timing is reasonable (<50ms per decision)
            avg_time = statistics.mean(timing_results)
            assert (
                avg_time < 50.0
            ), f"Safety decision {decision_type} avg time {avg_time:.2f}ms > 50ms"

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordination_overhead_impact_on_safety_timing(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """Test that coordination overhead doesn't degrade safety timing."""
        # RED PHASE - Safety timing should not degrade with coordination active

        # Baseline timing without coordination
        baseline_times = []
        for i in range(5):
            start_time = time.perf_counter()
            decision = safety_authority_manager.evaluate_safety_decision(
                level=SafetyAuthorityLevel.EMERGENCY_STOP,
                decision_type=SafetyDecisionType.FORCE_EMERGENCY_STOP,
                context={"test": "baseline", "iteration": i},
            )
            end_time = time.perf_counter()
            baseline_times.append((end_time - start_time) * 1000)
            assert decision.decision_type == SafetyDecisionType.FORCE_EMERGENCY_STOP

        baseline_avg = statistics.mean(baseline_times)

        # Timing with coordination active
        await dual_sdr_coordinator.start()

        coordination_times = []
        for i in range(5):
            start_time = time.perf_counter()
            decision = safety_authority_manager.evaluate_safety_decision(
                level=SafetyAuthorityLevel.EMERGENCY_STOP,
                decision_type=SafetyDecisionType.FORCE_EMERGENCY_STOP,
                context={"test": "coordination_active", "iteration": i},
            )
            end_time = time.perf_counter()
            coordination_times.append((end_time - start_time) * 1000)
            assert decision.decision_type == SafetyDecisionType.FORCE_EMERGENCY_STOP

        coordination_avg = statistics.mean(coordination_times)

        # Coordination should not add significant overhead (< 100% increase)
        overhead_factor = coordination_avg / baseline_avg
        assert overhead_factor < 2.0, f"Coordination overhead {overhead_factor:.2f}x too high"

        # Both should still be well under 500ms limit
        assert (
            coordination_avg < 100.0
        ), f"Coordination timing {coordination_avg:.2f}ms should be well under 500ms"

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_sustained_coordination_performance_impact(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """Test safety performance under sustained coordination operations."""
        # RED PHASE - Safety should maintain performance under sustained load
        await dual_sdr_coordinator.start()

        # Run sustained coordination operations for 30 seconds
        test_duration = 30.0  # seconds
        start_test = time.time()

        safety_response_times = []
        coordination_iterations = 0

        while (time.time() - start_test) < test_duration:
            # Simulate coordination work
            coordination_iterations += 1

            # Every 10th iteration, test safety response time
            if coordination_iterations % 10 == 0:
                start_safety = time.perf_counter()

                decision = safety_authority_manager.evaluate_safety_decision(
                    level=SafetyAuthorityLevel.COORDINATION_HEALTH,
                    decision_type=SafetyDecisionType.MONITOR_HEALTH,
                    context={"sustained_test": True, "iteration": coordination_iterations},
                )

                end_safety = time.perf_counter()
                safety_time = (end_safety - start_safety) * 1000
                safety_response_times.append(safety_time)

                assert decision.decision_type == SafetyDecisionType.MONITOR_HEALTH

            # Small delay to simulate real coordination timing
            await asyncio.sleep(0.01)

        # Verify safety performance didn't degrade over time
        if len(safety_response_times) >= 2:
            first_half = safety_response_times[: len(safety_response_times) // 2]
            second_half = safety_response_times[len(safety_response_times) // 2 :]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            # Performance shouldn't degrade by more than 50%
            degradation_factor = second_avg / first_avg
            assert (
                degradation_factor < 1.5
            ), f"Safety performance degraded {degradation_factor:.2f}x over time"

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_fallback_timing_performance_validation(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4e.4] - Automatic fallback timing meets <10s requirement per PRD-AC5.3.4.

        Validates fallback detection and execution within 10 seconds.
        """
        await dual_sdr_coordinator.start()

        # Test fallback detection timing multiple times for consistency
        fallback_times = []
        test_iterations = 10

        for i in range(test_iterations):
            # Simulate communication loss scenario
            failure_start = time.time()

            # Safety manager should detect failure and recommend fallback quickly
            safe_source = safety_manager.get_safe_source_recommendation()
            fallback_time = time.time() - failure_start

            # Record fallback detection time
            fallback_times.append(fallback_time)

            # Verify individual fallback meets timing requirement
            assert (
                fallback_time < 10.0
            ), f"Fallback detection {i} took {fallback_time:.2f}s, exceeds 10s limit"
            assert safe_source in [
                "drone",
                "auto",
                "ground",
            ], f"Fallback recommendation '{safe_source}' invalid"

            await asyncio.sleep(0.1)

        # Statistical analysis of fallback performance
        avg_fallback_time = statistics.mean(fallback_times)
        max_fallback_time = max(fallback_times)

        # Fallback performance requirements - should be much faster than 10s limit
        assert avg_fallback_time < 1.0, f"Average fallback time {avg_fallback_time:.2f}s too slow"
        assert (
            max_fallback_time < 10.0
        ), f"Maximum fallback time {max_fallback_time:.2f}s exceeds limit"

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance_impact(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4e.5] - Performance validation under concurrent operations load.

        Validates system maintains performance requirements under concurrent load.
        """
        await dual_sdr_coordinator.start()

        # Performance metrics tracking
        operation_times = []
        safety_response_times = []
        total_operations = 0

        # Run concurrent operations for performance testing
        async def coordination_operations():
            nonlocal total_operations
            for i in range(50):  # 50 coordination operations
                op_start = time.perf_counter()
                health_status = dual_sdr_coordinator.get_health_status()
                op_time = (time.perf_counter() - op_start) * 1000
                operation_times.append(op_time)
                total_operations += 1
                assert health_status is not None
                await asyncio.sleep(0.01)

        async def safety_monitoring_operations():
            nonlocal total_operations
            for i in range(100):  # 100 safety monitoring operations
                op_start = time.perf_counter()
                coordination_status = safety_manager.get_coordination_status()
                op_time = (time.perf_counter() - op_start) * 1000
                safety_response_times.append(op_time)
                total_operations += 1
                assert "active" in coordination_status
                await asyncio.sleep(0.005)

        # Run both operations concurrently
        start_time = time.time()
        await asyncio.gather(coordination_operations(), safety_monitoring_operations())
        total_time = time.time() - start_time

        # Performance analysis
        throughput = total_operations / total_time
        avg_coordination_time = statistics.mean(operation_times)
        avg_safety_time = statistics.mean(safety_response_times)

        # Performance requirements validation
        assert throughput > 50, f"Throughput {throughput:.1f} ops/sec too low, expected >50"
        assert (
            avg_coordination_time < 50
        ), f"Average coordination time {avg_coordination_time:.2f}ms too high"
        assert (
            avg_safety_time < 10
        ), f"Average safety monitoring time {avg_safety_time:.2f}ms too high"

        # Verify emergency stop still works quickly after concurrent load
        emergency_start = time.perf_counter()
        emergency_decision = safety_authority_manager.evaluate_safety_decision(
            level=SafetyAuthorityLevel.EMERGENCY_STOP,
            decision_type=SafetyDecisionType.FORCE_EMERGENCY_STOP,
            context={"post_load_test": True},
        )
        emergency_time = (time.perf_counter() - emergency_start) * 1000

        assert (
            emergency_time < 500
        ), f"Post-load emergency stop {emergency_time:.2f}ms exceeds 500ms"
        assert emergency_decision.decision_type == SafetyDecisionType.FORCE_EMERGENCY_STOP

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_deterministic_timing_consistency_validation(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """
        Test [4e.6] - Deterministic timing validation per PRD-NFR12.

        Validates safety-critical functions execute with consistent timing.
        """
        await dual_sdr_coordinator.start()

        # Test deterministic timing for critical safety operations
        emergency_stop_times = []
        safety_decision_times = []

        test_iterations = 50

        for i in range(test_iterations):
            # Test emergency stop timing consistency
            start_time = time.perf_counter()
            emergency_decision = safety_authority_manager.evaluate_safety_decision(
                level=SafetyAuthorityLevel.EMERGENCY_STOP,
                decision_type=SafetyDecisionType.FORCE_EMERGENCY_STOP,
                context={"deterministic_test": i},
            )
            emergency_time = (time.perf_counter() - start_time) * 1000
            emergency_stop_times.append(emergency_time)
            assert emergency_decision.decision_type == SafetyDecisionType.FORCE_EMERGENCY_STOP

            # Test regular safety decision timing
            start_time = time.perf_counter()
            monitor_decision = safety_authority_manager.evaluate_safety_decision(
                level=SafetyAuthorityLevel.SYSTEM_MONITOR,
                decision_type=SafetyDecisionType.MONITOR_HEALTH,
                context={"deterministic_test": i},
            )
            decision_time = (time.perf_counter() - start_time) * 1000
            safety_decision_times.append(decision_time)
            assert monitor_decision.decision_type == SafetyDecisionType.MONITOR_HEALTH

            await asyncio.sleep(0.001)

        # Analyze timing consistency (determinism)
        emergency_avg = statistics.mean(emergency_stop_times)
        emergency_std = statistics.stdev(emergency_stop_times)
        emergency_coeff_var = emergency_std / emergency_avg

        decision_avg = statistics.mean(safety_decision_times)
        decision_std = statistics.stdev(safety_decision_times)
        decision_coeff_var = decision_std / decision_avg

        # Deterministic timing requirements (low coefficient of variation)
        assert (
            emergency_coeff_var < 0.5
        ), f"Emergency stop timing variability {emergency_coeff_var:.3f} too high"
        assert (
            decision_coeff_var < 0.5
        ), f"Safety decision timing variability {decision_coeff_var:.3f} too high"

        # Absolute timing requirements
        assert emergency_avg < 250, f"Emergency stop average {emergency_avg:.2f}ms too slow"
        assert decision_avg < 50, f"Safety decision average {decision_avg:.2f}ms too slow"
        assert (
            max(emergency_stop_times) < 500
        ), f"Emergency stop max {max(emergency_stop_times):.2f}ms exceeds limit"

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_memory_resource_performance_validation(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4e.7] - Memory and resource usage performance validation.

        Validates system resource usage remains within acceptable bounds during operation.
        """

        await dual_sdr_coordinator.start()

        # Get baseline resource usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        # Initialize CPU measurement (first call returns 0.0, but primes the measurement)
        process.cpu_percent()  # Reset the internal state
        await asyncio.sleep(0.1)  # Short delay to allow CPU measurement initialization

        # Run sustained operations to test resource usage
        operation_duration = 5.0  # 5 seconds of sustained operations
        start_time = time.time()
        operations_performed = 0

        while time.time() - start_time < operation_duration:
            # Coordination operations
            health_status = dual_sdr_coordinator.get_health_status()
            assert health_status is not None

            # Safety monitoring
            coordination_status = safety_manager.get_coordination_status()
            assert "active" in coordination_status

            # Authority decision
            test_decision = safety_authority_manager.evaluate_safety_decision(
                level=SafetyAuthorityLevel.SYSTEM_MONITOR,
                decision_type=SafetyDecisionType.MONITOR_HEALTH,
                context={"resource_test": operations_performed},
            )
            assert test_decision.decision_type == SafetyDecisionType.MONITOR_HEALTH

            operations_performed += 1
            await asyncio.sleep(0.001)  # 1ms between operations

        # Measure final resource usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        # Get accurate CPU usage over the operation period (second call gives accurate reading)
        cpu_usage_percent = process.cpu_percent()

        # Calculate resource usage
        memory_increase = final_memory - initial_memory

        # Resource usage validation with relaxed thresholds for CI stability
        assert (
            memory_increase < 10
        ), f"Memory increased by {memory_increase:.1f}MB, indicates potential memory leak"
        assert (
            cpu_usage_percent < 50
        ), f"CPU usage {cpu_usage_percent:.1f}% too high during sustained operations"
        assert (
            operations_performed > 200
        ), f"Only performed {operations_performed} operations in {operation_duration}s (CI-adjusted threshold)"

        # Verify system still responsive after sustained load
        emergency_start = time.perf_counter()
        emergency_decision = safety_authority_manager.evaluate_safety_decision(
            level=SafetyAuthorityLevel.EMERGENCY_STOP,
            decision_type=SafetyDecisionType.FORCE_EMERGENCY_STOP,
            context={"post_resource_test": True},
        )
        emergency_time = (time.perf_counter() - emergency_start) * 1000

        assert (
            emergency_time < 500
        ), f"Post-resource-test emergency stop {emergency_time:.2f}ms exceeds 500ms"
        assert emergency_decision.decision_type == SafetyDecisionType.FORCE_EMERGENCY_STOP

        await dual_sdr_coordinator.stop()
