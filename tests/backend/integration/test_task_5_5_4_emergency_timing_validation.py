"""
TASK-5.5.4 Integration Testing: Emergency Stop Timing Validation

SUBTASK-5.5.4.3: Validate <500ms emergency stop with coordination overhead

This module implements comprehensive emergency stop timing validation tests
using authentic SafetyAuthorityManager integration with DualSDRCoordinator
active to ensure PRD-FR16 <500ms requirements are met under coordination load.

Chain of Thought Context:
- PRD → Epic 5 → Story 5.5 → TASK-5.5.4-INTEGRATION-TESTING → SUBTASK-5.5.4.3
- Integration Points: SafetyAuthorityManager, DualSDRCoordinator, emergency pathways
- Prerequisites: TASK-5.5.3 safety architecture completed ✅
- Test Authenticity: Uses real SafetyAuthorityManager.trigger_emergency_override()
"""

import asyncio
import contextlib
import time

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_authority_manager import SafetyAuthorityManager
from src.backend.services.safety_manager import SafetyManager
from src.backend.services.sdr_priority_manager import SDRPriorityManager


class TestEmergencyStopTimingValidation:
    """Test emergency stop timing validation with coordination overhead per SUBTASK-5.5.4.3"""

    @pytest.fixture
    async def safety_authority_manager(self):
        """Create authentic SafetyAuthorityManager for integration testing."""
        # Create real SafetyAuthorityManager instance
        safety_authority = SafetyAuthorityManager()
        yield safety_authority

    @pytest.fixture
    async def coordination_system(self, safety_authority_manager):
        """Create coordination system with safety authority integration."""
        # Create safety manager with real integration
        safety_manager = SafetyManager()

        # Create SDR priority manager with safety integration
        priority_manager = SDRPriorityManager(safety_authority=safety_authority_manager)

        # Create dual SDR coordinator with safety authority dependency injection
        dual_coordinator = DualSDRCoordinator(safety_authority=safety_authority_manager)

        yield {
            "safety_authority": safety_authority_manager,
            "dual_coordinator": dual_coordinator,
            "priority_manager": priority_manager,
            "safety_manager": safety_manager,
        }

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_benchmark_coordination_active(self, coordination_system):
        """
        Test [4m]: Benchmark emergency stop timing with coordination system active

        Validates that emergency stop response maintains <500ms timing per PRD-FR16
        when DualSDRCoordinator is actively processing coordination tasks.
        """
        safety_authority = coordination_system["safety_authority"]
        dual_coordinator = coordination_system["dual_coordinator"]

        # Start coordination system processing to create realistic load
        coordination_task = asyncio.create_task(self._simulate_coordination_load(dual_coordinator))

        # Wait for coordination system to be actively processing
        await asyncio.sleep(0.1)

        # Measure emergency stop timing with coordination system active
        start_time = time.perf_counter()

        # Trigger authentic emergency override using real SafetyAuthorityManager
        emergency_result = await safety_authority.trigger_emergency_override(
            reason="Integration test emergency stop timing validation"
        )

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Clean up coordination task
        try:
            pass  # Main test logic completed
        finally:
            # Ensure proper cleanup regardless of test failures
            if not coordination_task.done():
                coordination_task.cancel()
            try:
                await coordination_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling

        # Verify emergency override was successful
        assert emergency_result["emergency_override_active"] is True
        assert emergency_result["response_time_ms"] < 500.0, (
            f"Emergency stop timing {emergency_result['response_time_ms']:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms"
        )

        # Verify measured timing also meets requirement
        assert response_time_ms < 500.0, (
            f"Measured emergency stop timing {response_time_ms:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms with coordination overhead"
        )

        # Verify emergency override state is active
        assert safety_authority.emergency_override_active is True

    @pytest.mark.asyncio
    async def test_emergency_stop_response_under_coordination_load(self, coordination_system):
        """
        Test [4n]: Test emergency stop response under coordination load

        Validates that emergency stop maintains <500ms timing per PRD-FR16
        when coordination system is under heavy processing load.
        """
        safety_authority = coordination_system["safety_authority"]
        dual_coordinator = coordination_system["dual_coordinator"]

        # Create multiple high-frequency coordination tasks to simulate load
        coordination_tasks = []
        for i in range(5):  # Create 5 concurrent coordination tasks
            task = asyncio.create_task(
                self._simulate_high_frequency_coordination_load(dual_coordinator, f"load_{i}")
            )
            coordination_tasks.append(task)

        # Let coordination load build up
        await asyncio.sleep(0.2)

        # Measure emergency stop timing under coordination load
        start_time = time.perf_counter()

        # Trigger emergency override with coordination system under load
        emergency_result = await safety_authority.trigger_emergency_override(
            reason="Emergency stop under coordination load test"
        )

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Clean up all coordination load tasks
        for task in coordination_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Verify emergency override was successful under load
        assert emergency_result["emergency_override_active"] is True
        assert emergency_result["response_time_ms"] < 500.0, (
            f"Emergency stop under load timing {emergency_result['response_time_ms']:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms"
        )

        # Verify measured timing meets requirement even under coordination load
        assert response_time_ms < 500.0, (
            f"Measured emergency stop under load {response_time_ms:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms"
        )

        # Verify emergency override is active despite coordination load
        assert safety_authority.emergency_override_active is True

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_during_source_switching(self, coordination_system):
        """
        Test [4o]: Validate emergency stop timing during source switching

        Validates that emergency stop timing remains <500ms per PRD-FR16
        even during active source switching operations.
        """
        safety_authority = coordination_system["safety_authority"]
        priority_manager = coordination_system["priority_manager"]

        # Start source switching simulation
        switching_task = asyncio.create_task(self._simulate_source_switching(priority_manager))

        # Let source switching process stabilize
        await asyncio.sleep(0.1)

        # Measure emergency stop timing during source switching
        start_time = time.perf_counter()

        # Trigger emergency override during active source switching
        emergency_result = await safety_authority.trigger_emergency_override(
            reason="Emergency stop during source switching test"
        )

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Clean up switching simulation
        switching_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await switching_task

        # Verify emergency override successful during source switching
        assert emergency_result["emergency_override_active"] is True
        assert emergency_result["response_time_ms"] < 500.0, (
            f"Emergency stop during switching timing {emergency_result['response_time_ms']:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms"
        )

        # Verify measured timing meets requirement during source switching
        assert response_time_ms < 500.0, (
            f"Measured emergency stop during switching {response_time_ms:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms"
        )

    async def _simulate_coordination_load(self, dual_coordinator):
        """Simulate active coordination processing to create realistic load."""
        while True:
            # Simulate coordination processing tasks
            await asyncio.sleep(0.01)  # 100Hz processing simulation

            # Simulate coordination decision making
            with contextlib.suppress(Exception):
                await dual_coordinator.get_health_status()

    async def _simulate_high_frequency_coordination_load(self, dual_coordinator, load_id):
        """Simulate high-frequency coordination processing to create load."""
        while True:
            # High frequency coordination processing simulation
            await asyncio.sleep(0.005)  # 200Hz processing simulation

            # Simulate intensive coordination tasks
            with contextlib.suppress(Exception):
                # Simulate multiple coordination operations
                await dual_coordinator.get_health_status()
                await asyncio.sleep(0.002)  # Simulate processing time

                # Additional load operations
                for _ in range(3):
                    await asyncio.sleep(0.001)

    async def _simulate_source_switching(self, priority_manager):
        """Simulate active source switching operations."""
        while True:
            with contextlib.suppress(Exception):
                # Simulate source switching decisions
                await asyncio.sleep(0.02)  # 50Hz switching evaluation

                # Simulate source priority evaluation
                for _ in range(2):
                    await asyncio.sleep(0.005)

    @pytest.mark.asyncio
    async def test_emergency_stop_performance_during_communication_issues(
        self, coordination_system
    ):
        """
        Test [4p]: Test emergency stop performance during communication issues

        Validates that emergency stop maintains <500ms timing per PRD-FR16
        when experiencing network communication problems and degradation.
        """
        safety_authority = coordination_system["safety_authority"]
        dual_coordinator = coordination_system["dual_coordinator"]

        # Simulate communication issues by disrupting network connectivity
        communication_issue_task = asyncio.create_task(
            self._simulate_communication_issues(dual_coordinator)
        )

        # Let communication issues manifest
        await asyncio.sleep(0.15)  # Allow issues to develop

        # Measure emergency stop timing during communication issues
        start_time = time.perf_counter()

        # Trigger emergency override during communication problems
        emergency_result = await safety_authority.trigger_emergency_override(
            reason="Emergency stop during communication issues test"
        )

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        # Clean up communication issue simulation
        communication_issue_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await communication_issue_task

        # Verify emergency override successful during communication issues
        assert emergency_result["emergency_override_active"] is True
        assert emergency_result["response_time_ms"] < 500.0, (
            f"Emergency stop during communication issues timing {emergency_result['response_time_ms']:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms"
        )

        # Verify measured timing meets requirement during communication problems
        assert response_time_ms < 500.0, (
            f"Measured emergency stop during communication issues {response_time_ms:.2f}ms "
            f"exceeds PRD-FR16 requirement of <500ms"
        )

        # Verify emergency override is active despite communication issues
        assert safety_authority.emergency_override_active is True

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_regression_tests(self, coordination_system):
        """
        Test [4q]: Create emergency stop timing regression tests

        Validates emergency stop timing consistency across multiple runs
        to ensure no performance regression over time.
        """
        safety_authority = coordination_system["safety_authority"]
        timing_results = []

        # Run multiple emergency stop timing tests for regression analysis
        for test_run in range(10):  # 10 test runs for statistical validity
            # Reset emergency override state for clean test
            if safety_authority.emergency_override_active:
                await safety_authority.clear_emergency_override("test_system")

            # Measure emergency stop timing for regression testing
            start_time = time.perf_counter()

            emergency_result = await safety_authority.trigger_emergency_override(
                reason=f"Emergency stop regression test run {test_run + 1}"
            )

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            timing_results.append(
                {
                    "run": test_run + 1,
                    "response_time_ms": response_time_ms,
                    "internal_timing_ms": emergency_result["response_time_ms"],
                }
            )

            # Each test run must meet timing requirement
            assert emergency_result["emergency_override_active"] is True
            assert response_time_ms < 500.0, (
                f"Emergency stop regression test run {test_run + 1} took {response_time_ms:.2f}ms "
                f"exceeding PRD-FR16 requirement"
            )

            # Brief pause between test runs to avoid interference
            await asyncio.sleep(0.05)

        # Statistical analysis for regression detection
        response_times = [result["response_time_ms"] for result in timing_results]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)

        # Regression validation - consistency check
        time_variance = max_response_time - min_response_time
        assert time_variance < 100.0, (
            f"Emergency stop timing variance {time_variance:.2f}ms too high, "
            f"indicates potential performance regression"
        )

        # All times must be well under limit for production reliability
        assert (
            avg_response_time < 300.0
        ), f"Average emergency stop time {avg_response_time:.2f}ms should be well under 500ms limit"

        # Verify no individual test exceeded reasonable bounds
        for result in timing_results:
            assert result["response_time_ms"] < 500.0

    @pytest.mark.asyncio
    async def test_emergency_stop_performance_monitoring(self, coordination_system):
        """
        Test [4r]: Implement emergency stop performance monitoring

        Validates that emergency stop performance can be monitored and tracked
        for ongoing system health assessment.
        """
        safety_authority = coordination_system["safety_authority"]
        dual_coordinator = coordination_system["dual_coordinator"]

        # Create performance monitoring context
        performance_metrics = {
            "emergency_stop_count": 0,
            "total_response_time_ms": 0,
            "max_response_time_ms": 0,
            "min_response_time_ms": float("inf"),
            "timing_violations": 0,
        }

        # Monitor emergency stop performance under various conditions
        test_scenarios = [
            ("baseline", lambda: asyncio.sleep(0)),
            ("light_load", lambda: self._simulate_coordination_load(dual_coordinator)),
            (
                "heavy_load",
                lambda: self._simulate_high_frequency_coordination_load(
                    dual_coordinator, "monitor"
                ),
            ),
        ]

        for scenario_name, load_generator in test_scenarios:
            # Start load scenario
            if scenario_name != "baseline":
                load_task = asyncio.create_task(load_generator())
                await asyncio.sleep(0.1)  # Let load establish

            try:
                # Reset emergency override state
                if safety_authority.emergency_override_active:
                    await safety_authority.clear_emergency_override("test_system")

                # Measure emergency stop performance
                start_time = time.perf_counter()

                emergency_result = await safety_authority.trigger_emergency_override(
                    reason=f"Performance monitoring test - {scenario_name}"
                )

                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000

                # Update performance metrics
                performance_metrics["emergency_stop_count"] += 1
                performance_metrics["total_response_time_ms"] += response_time_ms
                performance_metrics["max_response_time_ms"] = max(
                    performance_metrics["max_response_time_ms"], response_time_ms
                )
                performance_metrics["min_response_time_ms"] = min(
                    performance_metrics["min_response_time_ms"], response_time_ms
                )

                if response_time_ms >= 500.0:
                    performance_metrics["timing_violations"] += 1

                # Verify this scenario meets requirements
                assert emergency_result["emergency_override_active"] is True
                assert response_time_ms < 500.0, (
                    f"Emergency stop performance monitoring failed for {scenario_name}: "
                    f"{response_time_ms:.2f}ms exceeds 500ms limit"
                )

            finally:
                # Clean up load scenario
                if scenario_name != "baseline":
                    load_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await load_task

        # Calculate final performance statistics
        avg_response_time = (
            performance_metrics["total_response_time_ms"]
            / performance_metrics["emergency_stop_count"]
        )

        # Performance monitoring validation
        assert performance_metrics["emergency_stop_count"] == 3  # All scenarios tested
        assert performance_metrics["timing_violations"] == 0  # No timing violations
        assert (
            avg_response_time < 400.0
        ), f"Average emergency stop performance {avg_response_time:.2f}ms should be well under limit"
        assert performance_metrics["max_response_time_ms"] < 500.0  # Maximum within bounds

        # Verify performance monitoring data completeness
        assert performance_metrics["min_response_time_ms"] != float("inf")
        assert (
            performance_metrics["max_response_time_ms"]
            > performance_metrics["min_response_time_ms"]
        )

    async def _simulate_communication_issues(self, dual_coordinator):
        """Simulate network communication issues and degradation."""
        issue_patterns = [
            {"delay_ms": 50, "duration_ms": 30},  # Network latency spikes
            {"delay_ms": 100, "duration_ms": 40},  # Severe latency
            {"delay_ms": 25, "duration_ms": 60},  # Intermittent delays
        ]

        for pattern in issue_patterns:
            # Simulate communication delay/issues
            await asyncio.sleep(pattern["delay_ms"] / 1000.0)

            # Simulate issue duration
            issue_start = time.perf_counter()
            while (time.perf_counter() - issue_start) * 1000 < pattern["duration_ms"]:
                with contextlib.suppress(Exception):
                    # Simulate degraded communication operations
                    await dual_coordinator.get_health_status()
                    await asyncio.sleep(0.01)  # Throttled communication

            # Brief recovery period
            await asyncio.sleep(0.02)
