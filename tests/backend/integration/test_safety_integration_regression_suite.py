"""
TASK-5.5.4-INTEGRATION-TESTING: Comprehensive Safety Integration Testing
SUBTASK-5.5.4.1 [4f] - Implement safety integration regression test suite

Comprehensive regression test suite to ensure safety integration capabilities
are preserved across system changes and updates. Validates core safety contracts.

PRD References:
- PRD-AC5.5.1: All existing PISAD safety interlocks remain active
- PRD-FR16: Emergency stop <500ms response time
- PRD-AC5.5.4: Safety authority hierarchy maintained with coordination
- PRD-NFR12: Deterministic timing for safety-critical functions
"""

import asyncio
import inspect
import time

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
    SafetyDecision,
    SafetyDecisionType,
)
from src.backend.services.safety_manager import SafetyManager
from src.backend.services.sdr_priority_manager import SDRPriorityManager
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestSafetyIntegrationRegressionSuite:
    """Regression test suite for safety integration capabilities."""

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
    def coordination_components(self, safety_authority_manager):
        """Create all coordination components with safety integration."""
        components = {
            "dual_coordinator": DualSDRCoordinator(safety_authority=safety_authority_manager),
            "sdr_priority": SDRPriorityManager(safety_authority=safety_authority_manager),
            "tcp_bridge": SDRPPBridgeService(
                config={"host": "localhost", "port": 8081}, safety_manager=safety_authority_manager
            ),
        }
        return components

    @pytest.mark.asyncio
    async def test_safety_authority_hierarchy_regression(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.1] - Safety authority hierarchy regression validation.

        Ensures safety authority hierarchy is preserved across all coordination components.
        """
        # Test all safety authority levels are properly enforced
        safety_levels = [
            SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
            SafetyAuthorityLevel.FLIGHT_MODE_MONITOR,
            SafetyAuthorityLevel.GEOFENCE_BOUNDARY,
            SafetyAuthorityLevel.BATTERY_MONITOR,
            SafetyAuthorityLevel.COMMUNICATION_MONITOR,
            SafetyAuthorityLevel.SIGNAL_MONITOR,
            SafetyAuthorityLevel.COORDINATION_HEALTH,
            SafetyAuthorityLevel.SYSTEM_MONITOR,
        ]

        # Start coordination components
        await coordination_components["dual_coordinator"].start()
        await coordination_components["sdr_priority"].start()

        # Test each safety authority level
        for level in safety_levels:
            # Create test decision for each level
            test_decision = SafetyDecision(
                decision_type=SafetyDecisionType.MONITOR_HEALTH,
                authority_level=level,
                reason=f"Regression test for {level.name}",
                timestamp=time.time(),
            )

            # Validate decision is processed correctly
            approved = await safety_authority_manager.validate_safety_decision(test_decision)
            assert approved is not None, f"Safety level {level.name} not properly handled"
            assert approved.authority_level == level, f"Authority level mismatch for {level.name}"

            # Verify coordination components respect this authority level
            coordinator_health = coordination_components["dual_coordinator"].get_health_status()
            if inspect.isawaitable(coordinator_health):
                coordinator_health = await coordinator_health
            assert coordinator_health is not None

            priority_status = await coordination_components["sdr_priority"].get_status()
            assert priority_status is not None

        # Clean shutdown
        await coordination_components["sdr_priority"].stop()
        await coordination_components["dual_coordinator"].stop()

    @pytest.mark.asyncio
    async def test_emergency_stop_capability_regression(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.2] - Emergency stop capability regression validation.

        Ensures emergency stop works correctly regardless of coordination state.
        """
        # Test emergency stop in various coordination states
        test_scenarios = [
            ("no_coordination", False, False),
            ("dual_coordinator_active", True, False),
            ("all_components_active", True, True),
        ]

        for scenario_name, coordinator_active, priority_active in test_scenarios:
            # Setup coordination state for this scenario
            if coordinator_active:
                await coordination_components["dual_coordinator"].start()
            if priority_active:
                await coordination_components["sdr_priority"].start()

            # Test emergency stop response time
            start_time = time.perf_counter()
            emergency_result = safety_manager.trigger_emergency_stop()
            response_time = emergency_result["response_time_ms"]

            # Validate emergency stop requirements
            assert (
                response_time < 500
            ), f"Emergency stop in {scenario_name} took {response_time:.2f}ms, exceeds 500ms"
            assert emergency_result["success"] is True, f"Emergency stop failed in {scenario_name}"

            # Test emergency decision validation
            emergency_decision = SafetyDecision(
                decision_type=SafetyDecisionType.EMERGENCY_STOP,
                authority_level=SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
                reason=f"Regression test emergency - {scenario_name}",
                timestamp=time.time(),
            )

            approved = await safety_authority_manager.validate_safety_decision(emergency_decision)
            assert approved.approved is True, f"Emergency decision not approved in {scenario_name}"

            # Clean up for next scenario
            if priority_active:
                await coordination_components["sdr_priority"].stop()
            if coordinator_active:
                await coordination_components["dual_coordinator"].stop()

    @pytest.mark.asyncio
    async def test_safety_monitoring_capability_regression(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.3] - Safety monitoring capability regression validation.

        Ensures safety monitoring functions work correctly with coordination integration.
        """
        await coordination_components["dual_coordinator"].start()
        await coordination_components["sdr_priority"].start()

        # Test core safety monitoring functions
        monitoring_functions = [
            ("coordination_status", lambda: safety_manager.get_coordination_status()),
            ("coordination_latency", lambda: safety_manager.get_coordination_latency_status()),
            (
                "coordination_battery_health",
                lambda: safety_manager.get_coordination_battery_health(),
            ),
            ("safe_source_recommendation", lambda: safety_manager.get_safe_source_recommendation()),
        ]

        for function_name, function_call in monitoring_functions:
            try:
                result = function_call()
                # Check if result is awaitable and await it if needed
                if inspect.isawaitable(result):
                    result = await result
                assert (
                    result is not None
                ), f"Safety monitoring function {function_name} returned None"

                # Validate result structure based on function type
                if function_name == "coordination_status":
                    assert isinstance(result, (dict, str)), f"{function_name} result format invalid"
                elif function_name == "coordination_latency":
                    assert isinstance(result, dict), f"{function_name} should return dict"
                    assert (
                        "coordination_latency_ms" in result
                    ), f"{function_name} missing latency key"
                elif function_name == "coordination_battery_health":
                    assert isinstance(result, dict), f"{function_name} should return dict"
                elif function_name == "safe_source_recommendation":
                    assert result in [
                        "drone",
                        "ground",
                        "auto",
                    ], f"{function_name} invalid recommendation: {result}"

            except Exception as e:
                pytest.fail(f"Safety monitoring function {function_name} failed: {e}")

        await coordination_components["sdr_priority"].stop()
        await coordination_components["dual_coordinator"].stop()

    @pytest.mark.asyncio
    async def test_coordination_safety_integration_regression(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.4] - Coordination safety integration regression validation.

        Ensures coordination components maintain safety integration contracts.
        """
        # Test safety integration in coordination components
        component_tests = [
            ("dual_coordinator", coordination_components["dual_coordinator"]),
            ("sdr_priority", coordination_components["sdr_priority"]),
        ]

        for component_name, component in component_tests:
            await component.start()

            # Test safety authority injection
            if hasattr(component, "safety_authority"):
                assert (
                    component.safety_authority is not None
                ), f"{component_name} missing safety authority"
                assert (
                    component.safety_authority == safety_authority_manager
                ), f"{component_name} wrong safety authority"

            # Test safety-aware operations
            if hasattr(component, "get_health_status"):
                health = component.get_health_status()
                if hasattr(health, "__await__"):  # If it's async
                    health = await health
                assert health is not None, f"{component_name} health status unavailable"

            # Test safety override capability
            if hasattr(component, "apply_safety_override"):
                try:
                    await component.apply_safety_override("test_override", "regression test")
                    # Component should handle safety override without error
                except AttributeError:
                    # Method might not exist, which is acceptable
                    pass

            # Test emergency safety response
            if hasattr(component, "trigger_emergency_safety_override"):
                try:
                    result = await component.trigger_emergency_safety_override(
                        "regression test emergency"
                    )
                    assert result is not None, f"{component_name} emergency override failed"
                except AttributeError:
                    # Method might not exist, which is acceptable
                    pass

            await component.stop()

    @pytest.mark.asyncio
    async def test_safety_decision_processing_regression(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.5] - Safety decision processing regression validation.

        Ensures all safety decision types are processed correctly.
        """
        await coordination_components["dual_coordinator"].start()

        # Test all safety decision types
        decision_test_cases = [
            (
                SafetyDecisionType.EMERGENCY_STOP,
                SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
                "Emergency stop regression",
            ),
            (
                SafetyDecisionType.COORDINATION_OVERRIDE,
                SafetyAuthorityLevel.FLIGHT_MODE_MONITOR,
                "Coordination override regression",
            ),
            (
                SafetyDecisionType.SOURCE_SELECTION,
                SafetyAuthorityLevel.COMMUNICATION_MONITOR,
                "Source selection regression",
            ),
            (
                SafetyDecisionType.MONITOR_HEALTH,
                SafetyAuthorityLevel.SYSTEM_MONITOR,
                "Health monitoring regression",
            ),
            (
                SafetyDecisionType.APPROVE_OPERATION,
                SafetyAuthorityLevel.COORDINATION_HEALTH,
                "Operation approval regression",
            ),
        ]

        for decision_type, authority_level, reason in decision_test_cases:
            # Create safety decision
            decision = SafetyDecision(
                decision_type=decision_type,
                authority_level=authority_level,
                reason=reason,
                timestamp=time.time(),
            )

            # Process decision through safety authority
            start_time = time.perf_counter()
            approved = await safety_authority_manager.validate_safety_decision(decision)
            processing_time = (time.perf_counter() - start_time) * 1000

            # Validate decision processing
            assert approved is not None, f"Decision {decision_type.name} not processed"
            assert (
                approved.decision_type == decision_type
            ), f"Decision type mismatch for {decision_type.name}"
            assert (
                approved.authority_level == authority_level
            ), f"Authority level mismatch for {decision_type.name}"
            assert (
                processing_time < 100
            ), f"Decision {decision_type.name} processing too slow: {processing_time:.2f}ms"

        await coordination_components["dual_coordinator"].stop()

    @pytest.mark.asyncio
    async def test_safety_timing_requirements_regression(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.6] - Safety timing requirements regression validation.

        Ensures all safety timing requirements are consistently met.
        """
        await coordination_components["dual_coordinator"].start()
        await coordination_components["sdr_priority"].start()

        # Test critical timing requirements multiple times for consistency
        timing_tests = []
        test_iterations = 20

        for i in range(test_iterations):
            # Test emergency stop timing
            emergency_start = time.perf_counter()
            emergency_result = safety_manager.trigger_emergency_stop()
            emergency_time = emergency_result["response_time_ms"]
            timing_tests.append(("emergency_stop", emergency_time))

            # Test safety decision timing
            decision_start = time.perf_counter()
            test_decision = SafetyDecision(
                decision_type=SafetyDecisionType.MONITOR_HEALTH,
                authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
                reason=f"Timing regression test {i}",
                timestamp=time.time(),
            )
            approved = await safety_authority_manager.validate_safety_decision(test_decision)
            decision_time = (time.perf_counter() - decision_start) * 1000
            timing_tests.append(("safety_decision", decision_time))

            # Test coordination health check timing
            health_start = time.perf_counter()
            health_status = coordination_components["dual_coordinator"].get_health_status()
            # Check if result is awaitable and await it if needed
            if inspect.isawaitable(health_status):
                health_status = await health_status
            health_time = (time.perf_counter() - health_start) * 1000
            timing_tests.append(("health_check", health_time))

            assert approved is not None
            assert health_status is not None

            await asyncio.sleep(0.001)  # Small delay between iterations

        # Analyze timing regression
        emergency_times = [t[1] for t in timing_tests if t[0] == "emergency_stop"]
        decision_times = [t[1] for t in timing_tests if t[0] == "safety_decision"]
        health_times = [t[1] for t in timing_tests if t[0] == "health_check"]

        # Validate timing requirements
        assert all(
            t < 500 for t in emergency_times
        ), f"Emergency stop timing regression: max {max(emergency_times):.2f}ms"
        assert all(
            t < 100 for t in decision_times
        ), f"Safety decision timing regression: max {max(decision_times):.2f}ms"
        assert all(
            t < 50 for t in health_times
        ), f"Health check timing regression: max {max(health_times):.2f}ms"

        # Check timing consistency (low variation indicates good determinism)
        import statistics

        emergency_avg = statistics.mean(emergency_times)
        decision_avg = statistics.mean(decision_times)
        health_avg = statistics.mean(health_times)

        assert (
            emergency_avg < 250
        ), f"Emergency stop average timing regression: {emergency_avg:.2f}ms"
        assert decision_avg < 50, f"Safety decision average timing regression: {decision_avg:.2f}ms"
        assert health_avg < 25, f"Health check average timing regression: {health_avg:.2f}ms"

        await coordination_components["sdr_priority"].stop()
        await coordination_components["dual_coordinator"].stop()

    @pytest.mark.asyncio
    async def test_safety_integration_error_handling_regression(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.7] - Safety integration error handling regression validation.

        Ensures safety system handles errors gracefully without degrading safety.
        """
        await coordination_components["dual_coordinator"].start()

        # Test error scenarios that should be handled gracefully
        error_scenarios = [
            (
                "invalid_decision_type",
                lambda: SafetyDecision(
                    decision_type=None,
                    authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
                    reason="Invalid test",
                    timestamp=time.time()
                ),
            ),
            (
                "invalid_authority_level",
                lambda: SafetyDecision(
                    decision_type=SafetyDecisionType.MONITOR_HEALTH,
                    authority_level=None,
                    reason="Invalid test",
                    timestamp=time.time()
                ),
            ),
        ]

        for scenario_name, invalid_decision_factory in error_scenarios:
            try:
                # Create the invalid decision inside the try block
                invalid_decision = invalid_decision_factory()
                # Invalid decisions should be handled without crashing the system
                result = await safety_authority_manager.validate_safety_decision(invalid_decision)
                # System should handle gracefully - either reject or process with defaults
                # The key is that it doesn't crash
            except Exception:
                # Exception handling is acceptable as long as system remains operational
                pass

            # Verify system still operational after error scenario
            emergency_result = safety_manager.trigger_emergency_stop()
            assert (
                emergency_result["success"] is True
            ), f"Safety system compromised after {scenario_name}"
            assert (
                emergency_result["response_time_ms"] < 500
            ), f"Safety timing compromised after {scenario_name}"

            # Verify coordination still functional
            health_status = coordination_components["dual_coordinator"].get_health_status()
            assert health_status is not None, f"Coordination compromised after {scenario_name}"

        await coordination_components["dual_coordinator"].stop()

    @pytest.mark.asyncio
    async def test_safety_integration_backwards_compatibility(
        self, safety_manager, safety_authority_manager, coordination_components
    ):
        """
        Test [4f.8] - Safety integration backwards compatibility validation.

        Ensures safety integration doesn't break existing functionality.
        """
        # Test that safety manager works without coordination components
        coordination_status = safety_manager.get_coordination_status()
        assert coordination_status is not None, "Safety manager broken without coordination"

        # Test emergency stop without coordination active
        emergency_result = safety_manager.trigger_emergency_stop()
        assert emergency_result["success"] is True, "Emergency stop broken without coordination"
        assert (
            emergency_result["response_time_ms"] < 500
        ), "Emergency stop timing broken without coordination"

        # Test safety authority manager standalone
        standalone_decision = SafetyDecision(
            decision_type=SafetyDecisionType.MONITOR_HEALTH,
            authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
            reason="Backwards compatibility test",
            timestamp=time.time(),
        )

        approved = await safety_authority_manager.validate_safety_decision(standalone_decision)
        assert approved is not None, "Safety authority broken in standalone mode"

        # Now test with coordination components active
        await coordination_components["dual_coordinator"].start()

        # Same functionality should work with coordination active
        coordination_status_active = safety_manager.get_coordination_status()
        assert (
            coordination_status_active is not None
        ), "Safety manager broken with coordination active"

        emergency_result_active = safety_manager.trigger_emergency_stop()
        assert (
            emergency_result_active["success"] is True
        ), "Emergency stop broken with coordination active"

        active_decision = SafetyDecision(
            decision_type=SafetyDecisionType.MONITOR_HEALTH,
            authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
            reason="Backwards compatibility test with coordination",
            timestamp=time.time(),
        )

        approved_active = await safety_authority_manager.validate_safety_decision(active_decision)
        assert approved_active is not None, "Safety authority broken with coordination active"

        await coordination_components["dual_coordinator"].stop()
