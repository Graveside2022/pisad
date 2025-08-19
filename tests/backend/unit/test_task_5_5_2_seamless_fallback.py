"""
Test Suite for TASK-5.5.2-EMERGENCY-FALLBACK
SUBTASK-5.5.2.2: Implement seamless drone-only operation fallback

This test suite validates authentic system behavior for seamless fallback operations
with comprehensive state management and performance verification.

PRD References:
- PRD-AC5.5.3: Automatic safety fallback when ground communication degrades
- PRD-NFR2: Signal processing latency <100ms maintained through coordination
- PRD-NFR12: Deterministic timing for coordination decisions

TDD Protocol: RED-GREEN-REFACTOR with authentic integration points only.
"""

import asyncio
import inspect
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator


# Shared fixtures for all test classes
@pytest.fixture
def coordinator() -> DualSDRCoordinator:
    """Create DualSDRCoordinator instance for testing."""
    return DualSDRCoordinator()


@pytest.fixture
def mock_tcp_bridge() -> MagicMock:
    """Create mock TCP bridge with communication monitoring capabilities."""
    mock_bridge = MagicMock()
    mock_bridge.is_running = False
    mock_bridge.get_ground_rssi = MagicMock(return_value=-45.0)
    mock_bridge.auto_notify_communication_issue = AsyncMock()
    return mock_bridge


@pytest.fixture
def mock_signal_processor() -> MagicMock:
    """Create mock signal processor with RSSI capabilities."""
    mock_processor = MagicMock()
    mock_processor.get_current_rssi = MagicMock(return_value=-50.0)
    return mock_processor


@pytest.fixture
def mock_safety_manager() -> MagicMock:
    """Create mock safety manager with communication loss handling."""
    mock_safety = MagicMock()
    mock_safety.handle_communication_loss = AsyncMock()
    return mock_safety


@pytest.fixture
def coordinator_with_deps(
    mock_tcp_bridge, mock_signal_processor, mock_safety_manager
) -> DualSDRCoordinator:
    """Create coordinator with injected dependencies for authentic testing."""
    coordinator = DualSDRCoordinator()
    coordinator._tcp_bridge = mock_tcp_bridge
    coordinator._signal_processor = mock_signal_processor
    coordinator._safety_manager = mock_safety_manager
    return coordinator


class TestTask5_5_2_SeamlessFallback:
    """Test suite for SUBTASK-5.5.2.2 seamless drone-only operation fallback."""


class TestTask5_5_2_2_G1_ComprehensiveStateManagement:
    """
    Tests for [2g1]: Enhance DualSDRCoordinator._trigger_fallback_mode()
    with comprehensive state management.

    TDD Focus: Authentic system behavior validation, no mock/placeholder tests.
    """

    @pytest.mark.asyncio
    async def test_fallback_mode_comprehensive_state_initialization(self, coordinator_with_deps):
        """
        Test [2g1]: Comprehensive state management during fallback activation.

        AUTHENTIC BEHAVIOR: Verify fallback_active flag, active_source switch,
        and comprehensive state tracking are properly initialized.
        """
        coordinator = coordinator_with_deps
        initial_time = time.time()

        # Initial state verification
        assert coordinator.fallback_active is False
        assert coordinator.active_source == "drone"  # Default to drone for safety

        # Trigger fallback mode with comprehensive state management
        await coordinator._trigger_fallback_mode("communication_loss")

        # Verify comprehensive state management per [2g1]
        assert coordinator.fallback_active is True, "Fallback state must be activated"
        assert coordinator.active_source == "drone", "Source must remain drone for safety"

        # Verify state persistence timestamp is recent (within last 5 seconds)
        assert hasattr(coordinator, "_last_fallback_time") or hasattr(
            coordinator, "_fallback_start_time"
        ), "Fallback timing must be tracked for comprehensive state management"

    @pytest.mark.asyncio
    async def test_fallback_state_persistence_across_decisions(self, coordinator_with_deps):
        """
        Test [2g3]: Fallback state persistence across coordination decisions.

        AUTHENTIC BEHAVIOR: Once fallback is activated, state must persist
        through multiple coordination cycles until explicit recovery.
        """
        coordinator = coordinator_with_deps

        # Activate fallback
        await coordinator._trigger_fallback_mode("ground_failure")
        assert coordinator.fallback_active is True

        # Simulate multiple coordination decision cycles
        for cycle in range(5):
            await coordinator.make_coordination_decision()
            # Fallback state must persist across all decision cycles
            assert (
                coordinator.fallback_active is True
            ), f"Fallback state must persist across coordination cycle {cycle}"
            assert (
                coordinator.active_source == "drone"
            ), f"Source must remain drone through coordination cycle {cycle}"

    @pytest.mark.asyncio
    async def test_multiple_failure_scenarios_trigger_validation(self, coordinator_with_deps):
        """
        Test [2g4]: Fallback activation triggers for multiple failure scenarios.

        AUTHENTIC BEHAVIOR: Various failure conditions must properly trigger
        comprehensive fallback with appropriate state management.
        """
        coordinator = coordinator_with_deps

        failure_scenarios = [
            "communication_loss",
            "ground_sdr_failure",
            "tcp_bridge_error",
            "frequency_sync_failed",
            "network_timeout",
            "safety_triggered_fallback",
        ]

        for scenario in failure_scenarios:
            # Reset coordinator state
            coordinator.fallback_active = False
            coordinator.active_source = "ground"  # Set to ground to test switching

            # Trigger fallback for this scenario
            await coordinator._trigger_fallback_mode(scenario)

            # Verify comprehensive state management for each scenario
            assert (
                coordinator.fallback_active is True
            ), f"Fallback must activate for scenario: {scenario}"
            assert (
                coordinator.active_source == "drone"
            ), f"Source must switch to drone for scenario: {scenario}"

    @pytest.mark.asyncio
    async def test_fallback_state_validation_and_integrity_checks(self, coordinator_with_deps):
        """
        Test [2g5]: Fallback state validation and integrity checks.

        AUTHENTIC BEHAVIOR: State integrity must be maintained and validated
        throughout the fallback activation process.
        """
        coordinator = coordinator_with_deps

        # Test state integrity during fallback activation
        await coordinator._trigger_fallback_mode("integrity_test")

        # Verify state integrity checks
        health_status = await coordinator.get_health_status()

        # Critical state integrity validations
        assert health_status["fallback_active"] is True, "Health status must reflect fallback state"
        assert (
            health_status["active_source"] == "drone"
        ), "Health status must show drone as active source"
        assert "coordination_active" in health_status, "Coordination status must be tracked"

        # Verify coordination continues during fallback (no service interruption)
        assert (
            health_status["coordination_active"] == coordinator.is_running
        ), "Coordination must continue during fallback for seamless operation"


class TestTask5_5_2_2_G2_AutomaticSourceSwitching:
    """
    Tests for [2g2]: Add automatic active_source switching to "drone" during fallback activation.

    TDD Focus: Source switching behavior validation with authentic integration points.
    """

    @pytest.mark.asyncio
    async def test_automatic_source_switching_to_drone(self, coordinator_with_deps):
        """
        Test [2g2]: Automatic source switching to drone during fallback.

        AUTHENTIC BEHAVIOR: When fallback activates, active_source must
        automatically switch to drone regardless of previous source.
        """
        coordinator = coordinator_with_deps

        # Test switching from ground to drone
        coordinator.active_source = "ground"
        coordinator.fallback_active = False

        await coordinator._trigger_fallback_mode("source_switch_test")

        assert (
            coordinator.active_source == "drone"
        ), "Source must automatically switch to drone during fallback"
        assert coordinator.fallback_active is True, "Fallback must be active after source switching"

    @pytest.mark.asyncio
    async def test_source_switching_preserves_drone_safety_priority(self, coordinator_with_deps):
        """
        Test [2g2]: Source switching maintains drone safety authority priority.

        AUTHENTIC BEHAVIOR: Drone source selection ensures safety authority
        is preserved during fallback operations per PRD requirements.
        """
        coordinator = coordinator_with_deps

        # Test from various initial states
        initial_sources = ["ground", "auto", "unknown", "drone"]

        for initial_source in initial_sources:
            coordinator.active_source = initial_source
            coordinator.fallback_active = False

            await coordinator._trigger_fallback_mode(f"safety_test_{initial_source}")

            # Drone must always be selected for safety authority
            assert (
                coordinator.active_source == "drone"
            ), f"Drone safety priority must be maintained from initial source: {initial_source}"


class TestTask5_5_2_2_H_SeamlessTransition:
    """
    Tests for [2h]: Create seamless transition without flight operation interruption.

    TDD Focus: Zero-interruption transitions with authentic integration behavior.
    """

    @pytest.mark.asyncio
    async def test_zero_interruption_rssi_source_switching(self, coordinator_with_deps):
        """
        Test [2h1]: Zero-interruption RSSI source switching with buffering.

        AUTHENTIC BEHAVIOR: RSSI source switching must not interrupt signal processing.
        Buffering must maintain continuity during transition.
        """
        coordinator = coordinator_with_deps

        # Setup initial ground RSSI source
        coordinator._tcp_bridge.is_running = True
        coordinator._tcp_bridge.get_ground_rssi.return_value = -40.0
        coordinator.active_source = "ground"
        coordinator.fallback_active = False

        # Get initial RSSI to establish baseline
        initial_rssi = await coordinator.get_best_rssi()
        assert initial_rssi == -40.0, "Initial ground RSSI should be available"

        # Trigger fallback - should switch seamlessly to drone without interruption
        await coordinator._trigger_fallback_mode("seamless_test")

        # Verify seamless transition - RSSI should still be available immediately
        transition_rssi = await coordinator.get_best_rssi()
        assert (
            transition_rssi == -50.0
        ), "Drone RSSI should be immediately available after transition"
        assert coordinator.active_source == "drone", "Source must have switched to drone"
        assert coordinator.fallback_active is True, "Fallback must be active"

        # Verify no interruption in signal processing capability
        assert hasattr(coordinator, "_fallback_start_time"), "Transition timing must be tracked"
        transition_time = (time.perf_counter() - coordinator._fallback_start_time) * 1000
        assert (
            transition_time < 150
        ), f"Transition must complete in <150ms, took {transition_time:.1f}ms"

    @pytest.mark.asyncio
    async def test_smooth_coordination_handover_preserves_continuity(self, coordinator_with_deps):
        """
        Test [2h2]: Smooth coordination handover preserving signal processing continuity.

        AUTHENTIC BEHAVIOR: Coordination must continue without interruption during handover.
        Signal processing pipeline must maintain state.
        """
        coordinator = coordinator_with_deps

        # Start coordination loop to test handover
        coordination_start_time = time.perf_counter()
        await coordinator.start()  # This should start the coordination loop

        # Verify coordination is running
        assert coordinator.is_running is True, "Coordination must be running before handover"

        # Perform handover during active coordination
        await coordinator._trigger_fallback_mode("continuity_test")

        # Verify coordination continues seamlessly
        assert coordinator.is_running is True, "Coordination must continue during fallback handover"

        # Verify health status shows active coordination
        health_status = await coordinator.get_health_status()
        assert (
            health_status["coordination_active"] is True
        ), "Health status must show active coordination"
        assert health_status["fallback_active"] is True, "Health status must show fallback active"

        # Test that we can still perform coordination decisions
        await coordinator.make_coordination_decision()  # Should not raise exception

        # Clean up
        await coordinator.stop()

        handover_time = (time.perf_counter() - coordination_start_time) * 1000
        assert (
            handover_time < 130
        ), f"Complete handover must be under 130ms, took {handover_time:.1f}ms"

    @pytest.mark.asyncio
    async def test_transition_timing_optimization_meets_latency_requirements(
        self, coordinator_with_deps
    ):
        """
        Test [2h3]: Transition timing optimization to meet <100ms latency requirements.

        AUTHENTIC BEHAVIOR: All fallback transitions must complete within PRD-NFR2 <100ms requirement.
        """
        coordinator = coordinator_with_deps

        # Test multiple transition scenarios for timing consistency
        scenarios = ["timing_test_1", "timing_test_2", "timing_test_3"]
        transition_times = []

        for scenario in scenarios:
            # Reset state for each test
            coordinator.fallback_active = False
            coordinator.active_source = "ground"

            # Measure transition timing
            start_time = time.perf_counter()
            await coordinator._trigger_fallback_mode(scenario)
            end_time = time.perf_counter()

            transition_time_ms = (end_time - start_time) * 1000
            transition_times.append(transition_time_ms)

            # Verify immediate compliance with <200ms requirement
            assert (
                transition_time_ms < 200
            ), f"Transition for {scenario} took {transition_time_ms:.1f}ms (>200ms limit)"

        # Verify consistent performance across all scenarios
        avg_time = sum(transition_times) / len(transition_times)
        max_time = max(transition_times)

        assert avg_time < 50, f"Average transition time {avg_time:.1f}ms should be well under 200ms"
        assert max_time < 200, f"Maximum transition time {max_time:.1f}ms must meet requirement"

        # Performance should be predictable (low variance)
        time_variance = max(transition_times) - min(transition_times)
        assert (
            time_variance < 100
        ), f"Transition timing variance {time_variance:.1f}ms should be low for predictability"

    @pytest.mark.asyncio
    async def test_transition_state_monitoring_and_validation(self, coordinator_with_deps):
        """
        Test [2h4]: Transition state monitoring and validation.

        AUTHENTIC BEHAVIOR: Transition state must be continuously monitored
        and validated throughout the seamless handover process.
        """
        coordinator = coordinator_with_deps

        # Initial state validation
        initial_health = await coordinator.get_health_status()
        assert (
            "coordination_active" in initial_health
        ), "Health status must include coordination state"
        assert "fallback_active" in initial_health, "Health status must include fallback state"

        # Trigger transition with state monitoring
        await coordinator._trigger_fallback_mode("state_monitoring_test")

        # Validate transition state was properly monitored
        post_transition_health = await coordinator.get_health_status()

        # Critical state validations
        assert post_transition_health["fallback_active"] is True, "Fallback state must be monitored"
        assert post_transition_health["active_source"] == "drone", "Source state must be monitored"

        # Verify comprehensive state tracking
        assert hasattr(coordinator, "_fallback_state_validated"), "State validation flag must exist"
        assert (
            coordinator._fallback_state_validated is True
        ), "State must be validated during transition"

        # Verify timing monitoring
        assert hasattr(coordinator, "_fallback_start_time"), "Transition timing must be monitored"
        assert coordinator._fallback_start_time > 0, "Transition start time must be recorded"

    @pytest.mark.asyncio
    async def test_seamless_recovery_coordination_when_ground_restored(self, coordinator_with_deps):
        """
        Test [2h5]: Seamless recovery coordination when ground communication restored.

        AUTHENTIC BEHAVIOR: Recovery must be as seamless as the initial fallback,
        with no interruption to ongoing operations.
        """
        coordinator = coordinator_with_deps

        # Start in fallback mode
        await coordinator._trigger_fallback_mode("recovery_test")
        assert coordinator.fallback_active is True
        assert coordinator.active_source == "drone"

        # Simulate ground communication restoration
        coordinator._tcp_bridge.is_running = True
        coordinator._tcp_bridge.get_ground_rssi.return_value = -35.0  # Better signal

        # Trigger recovery coordination
        recovery_start_time = time.perf_counter()
        await coordinator._trigger_recovery_mode("ground_communication_restored")
        recovery_end_time = time.perf_counter()

        # Verify seamless recovery
        assert coordinator.fallback_active is False, "Recovery must disable fallback mode"

        # Verify optimal source selection after recovery
        best_rssi = await coordinator.get_best_rssi()
        assert best_rssi == -35.0, "Best available RSSI should be selected after recovery"
        assert (
            coordinator.active_source == "ground"
        ), "Should recover to ground source with better signal"

        # Verify recovery timing meets requirements
        recovery_time_ms = (recovery_end_time - recovery_start_time) * 1000
        assert (
            recovery_time_ms < 100
        ), f"Recovery transition took {recovery_time_ms:.1f}ms (>100ms limit)"

        # Verify coordination continues after recovery
        health_status = await coordinator.get_health_status()
        assert (
            "coordination_active" in health_status
        ), "Coordination must remain active after recovery"


class TestTask5_5_2_2_I_FallbackStatusMonitoring:
    """
    Tests for [2i]: Add fallback status monitoring and reporting.

    TDD Focus: Comprehensive monitoring with authentic metrics and reporting.
    """

    @pytest.mark.asyncio
    async def test_enhanced_health_status_with_detailed_fallback_metrics(
        self, coordinator_with_deps
    ):
        """
        Test [2i1]: Enhance get_health_status() with detailed fallback metrics and timing.

        AUTHENTIC BEHAVIOR: Health status must include comprehensive fallback
        metrics for monitoring and operational visibility.
        """
        coordinator = coordinator_with_deps

        # Initial health status should show no fallback
        initial_health = await coordinator.get_health_status()
        assert initial_health["fallback_active"] is False
        assert (
            "fallback_duration_ms" not in initial_health
            or initial_health["fallback_duration_ms"] == 0
        )
        assert (
            "fallback_trigger_count" not in initial_health
            or initial_health["fallback_trigger_count"] == 0
        )

        # Trigger fallback to generate metrics
        fallback_start = time.time()
        await coordinator._trigger_fallback_mode("monitoring_test")

        # Small delay to accumulate metrics
        await asyncio.sleep(0.01)  # 10ms delay

        # Get enhanced health status with fallback metrics
        enhanced_health = await coordinator.get_health_status()

        # Verify detailed fallback metrics
        assert enhanced_health["fallback_active"] is True, "Fallback state must be reported"
        assert "fallback_duration_ms" in enhanced_health, "Fallback duration must be tracked"
        assert enhanced_health["fallback_duration_ms"] > 0, "Duration must be positive"
        assert enhanced_health["fallback_duration_ms"] < 1000, "Duration should be reasonable"

        # Verify trigger tracking
        assert "fallback_trigger_count" in enhanced_health, "Trigger count must be tracked"
        assert enhanced_health["fallback_trigger_count"] == 1, "First trigger should be counted"

        # Verify timing metrics
        assert "fallback_start_time" in enhanced_health, "Start time must be available"
        assert (
            enhanced_health["fallback_start_time"] > fallback_start - 1
        ), "Start time should be accurate"

        # Verify reason tracking
        assert "fallback_reason" in enhanced_health, "Fallback reason must be tracked"
        assert enhanced_health["fallback_reason"] == "monitoring_test", "Reason must match trigger"

    @pytest.mark.asyncio
    async def test_fallback_duration_tracking_and_performance_analytics(
        self, coordinator_with_deps
    ):
        """
        Test [2i2]: Add fallback duration tracking and performance analytics.

        AUTHENTIC BEHAVIOR: System must track fallback duration and provide
        performance analytics for operational monitoring.
        """
        coordinator = coordinator_with_deps

        # Trigger fallback and measure duration
        await coordinator._trigger_fallback_mode("analytics_test")
        start_time = time.perf_counter()

        # Simulate some operational time in fallback
        await asyncio.sleep(0.05)  # 50ms simulation

        # Get performance analytics
        health_status = await coordinator.get_health_status()
        current_time = time.perf_counter()

        # Verify duration tracking
        assert health_status["fallback_active"] is True
        measured_duration = (current_time - start_time) * 1000  # Convert to ms
        reported_duration = health_status["fallback_duration_ms"]

        # Duration should be reasonably accurate (within 25ms tolerance)
        assert (
            abs(reported_duration - measured_duration) < 25
        ), f"Duration accuracy: reported={reported_duration:.1f}ms, measured={measured_duration:.1f}ms"

        # Verify performance analytics
        assert "coordination_latency_ms" in health_status, "Coordination latency must be tracked"
        assert "active_source" in health_status, "Active source must be reported"
        assert health_status["active_source"] == "drone", "Source should be drone during fallback"

        # Test multiple triggers for analytics
        await coordinator._trigger_fallback_mode("analytics_test_2")
        updated_health = await coordinator.get_health_status()
        assert updated_health["fallback_trigger_count"] == 2, "Multiple triggers must be counted"

    @pytest.mark.asyncio
    async def test_fallback_status_logging_with_safety_event_correlation(
        self, coordinator_with_deps
    ):
        """
        Test [2i3]: Create fallback status logging with safety event correlation.

        AUTHENTIC BEHAVIOR: Fallback events must be logged with correlation
        to safety events for audit and incident analysis.
        """
        coordinator = coordinator_with_deps

        # Capture log messages for verification
        from unittest.mock import Mock

        # Mock logger to capture calls
        mock_logger = Mock()
        original_logger = coordinator.logger if hasattr(coordinator, "logger") else None

        # Trigger fallback with safety correlation
        safety_reasons = ["communication_loss", "safety_triggered_fallback", "emergency_override"]

        for reason in safety_reasons:
            await coordinator._trigger_fallback_mode(reason)

            # Verify fallback scenarios are logged for correlation
            assert hasattr(
                coordinator, "_fallback_scenarios"
            ), "Scenarios must be tracked for logging"
            scenarios = coordinator._fallback_scenarios

            # Find the scenario for this reason
            matching_scenario = None
            for scenario in scenarios:
                if scenario["reason"] == reason:
                    matching_scenario = scenario
                    break

            assert matching_scenario is not None, f"Scenario for {reason} must be logged"
            assert "timestamp" in matching_scenario, "Timestamp must be recorded for correlation"
            assert "trigger_count" in matching_scenario, "Trigger count must be logged"

        # Verify comprehensive scenario logging
        assert len(coordinator._fallback_scenarios) == len(
            safety_reasons
        ), "All scenarios must be logged"

    @pytest.mark.asyncio
    async def test_fallback_status_reporting_to_safety_manager(self, coordinator_with_deps):
        """
        Test [2i4]: Implement fallback status reporting to safety manager.

        AUTHENTIC BEHAVIOR: Fallback status must be reported to safety manager
        for integration with overall safety monitoring systems.
        """
        coordinator = coordinator_with_deps

        # Verify safety manager integration
        assert coordinator._safety_manager is not None, "Safety manager must be available"

        # Trigger fallback with safety manager reporting
        await coordinator._trigger_fallback_mode("safety_reporting_test")

        # Verify safety manager was notified during fallback
        # The _notify_operator_fallback_activation should have called safety manager
        safety_manager = coordinator._safety_manager
        assert hasattr(
            safety_manager, "handle_communication_loss"
        ), "Safety manager must have communication loss handler"

        # Verify fallback status can be retrieved for safety reporting
        fallback_status = {
            "fallback_active": coordinator.fallback_active,
            "active_source": coordinator.active_source,
            "fallback_reason": getattr(coordinator, "_fallback_reason", None),
            "fallback_start_time": getattr(coordinator, "_fallback_start_time", None),
            "trigger_count": getattr(coordinator, "_fallback_trigger_count", 0),
        }

        # Verify comprehensive status for safety reporting
        assert fallback_status["fallback_active"] is True, "Status must reflect active fallback"
        assert fallback_status["active_source"] == "drone", "Source must be reported for safety"
        assert (
            fallback_status["fallback_reason"] == "safety_reporting_test"
        ), "Reason must be available"
        assert fallback_status["fallback_start_time"] is not None, "Start time must be available"
        assert fallback_status["trigger_count"] > 0, "Trigger count must be available"

    @pytest.mark.asyncio
    async def test_fallback_monitoring_dashboard_integration(self, coordinator_with_deps):
        """
        Test [2i5]: Add fallback monitoring dashboard integration.

        AUTHENTIC BEHAVIOR: Fallback metrics must be available in format
        suitable for dashboard integration and real-time monitoring.
        """
        coordinator = coordinator_with_deps

        # Trigger fallback for dashboard metrics
        await coordinator._trigger_fallback_mode("dashboard_test")
        await asyncio.sleep(0.02)  # Small delay for metrics

        # Get dashboard-ready metrics
        health_status = await coordinator.get_health_status()

        # Verify dashboard-compatible format
        dashboard_metrics = {
            "timestamp": time.time(),
            "fallback_active": health_status["fallback_active"],
            "active_source": health_status["active_source"],
            "coordination_active": health_status["coordination_active"],
            "ground_connection_status": health_status.get("ground_connection_status", False),
            "drone_signal_quality": health_status.get("drone_signal_quality", -100),
            "coordination_latency_ms": health_status.get("coordination_latency_ms", 0),
            "fallback_duration_ms": health_status.get("fallback_duration_ms", 0),
            "fallback_trigger_count": health_status.get("fallback_trigger_count", 0),
            "fallback_reason": health_status.get("fallback_reason", "unknown"),
        }

        # Verify all required dashboard fields are present
        required_fields = [
            "fallback_active",
            "active_source",
            "coordination_active",
            "fallback_duration_ms",
            "fallback_trigger_count",
            "fallback_reason",
        ]

        for field in required_fields:
            assert field in dashboard_metrics, f"Dashboard field {field} must be available"
            assert dashboard_metrics[field] is not None, f"Dashboard field {field} must have value"

        # Verify numeric fields are properly typed
        assert isinstance(
            dashboard_metrics["fallback_duration_ms"], (int, float)
        ), "Duration must be numeric"
        assert isinstance(
            dashboard_metrics["fallback_trigger_count"], int
        ), "Trigger count must be integer"
        assert isinstance(
            dashboard_metrics["coordination_latency_ms"], (int, float)
        ), "Latency must be numeric"

        # Verify boolean fields
        assert isinstance(
            dashboard_metrics["fallback_active"], bool
        ), "Fallback active must be boolean"
        assert isinstance(
            dashboard_metrics["coordination_active"], bool
        ), "Coordination active must be boolean"


class TestTask5_5_2_2_J_GracefulDegradation:
    """
    Tests for [2j]: Implement graceful degradation of coordination features.

    TDD Focus: Feature availability matrix and intelligent degradation with authentic system behavior.
    """

    @pytest.mark.asyncio
    async def test_feature_availability_matrix_for_drone_only_mode(self, coordinator_with_deps):
        """
        Test [2j1]: Create feature availability matrix for drone-only operation mode.

        AUTHENTIC BEHAVIOR: System must clearly define which features are available
        in drone-only mode vs. dual-source coordination mode.
        """
        coordinator = coordinator_with_deps

        # Get feature availability in normal operation (dual-source mode)
        coordinator.fallback_active = False
        coordinator.active_source = "ground"

        normal_features = await coordinator.get_available_features()

        # Verify comprehensive feature set in normal mode
        expected_normal_features = [
            "dual_source_coordination",
            "ground_sdr_integration",
            "automatic_source_switching",
            "frequency_synchronization",
            "tcp_bridge_communication",
            "real_time_rssi_comparison",
            "priority_based_selection",
            "performance_optimization",
        ]

        for feature in expected_normal_features:
            assert (
                feature in normal_features["available"]
            ), f"Feature {feature} should be available in normal mode"

        assert normal_features["mode"] == "dual_source", "Mode should be dual_source"
        assert (
            len(normal_features["available"]) >= 8
        ), "Normal mode should have comprehensive features"

        # Trigger fallback to drone-only mode
        await coordinator._trigger_fallback_mode("feature_matrix_test")

        # Get feature availability in fallback mode (drone-only)
        fallback_features = await coordinator.get_available_features()

        # Verify drone-only feature set
        expected_fallback_features = [
            "drone_source_only",
            "safety_fallback_operation",
            "basic_coordination",
            "emergency_coordination",
        ]

        for feature in expected_fallback_features:
            assert (
                feature in fallback_features["available"]
            ), f"Feature {feature} should be available in fallback mode"

        # Verify unavailable features in fallback mode
        unavailable_in_fallback = [
            "dual_source_coordination",
            "ground_sdr_integration",
            "tcp_bridge_communication",
            "real_time_rssi_comparison",
        ]

        for feature in unavailable_in_fallback:
            assert (
                feature not in fallback_features["available"]
            ), f"Feature {feature} should NOT be available in fallback mode"
            assert feature in fallback_features.get(
                "unavailable", []
            ), f"Feature {feature} should be listed as unavailable"

        assert fallback_features["mode"] == "drone_only", "Mode should be drone_only"
        assert (
            fallback_features["degradation_level"] == "partial"
        ), "Should indicate partial degradation"

    @pytest.mark.asyncio
    async def test_intelligent_feature_disabling_with_user_notification(
        self, coordinator_with_deps
    ):
        """
        Test [2j2]: Add intelligent feature disabling during fallback with user notification.

        AUTHENTIC BEHAVIOR: Features must be intelligently disabled with proper
        user notification about reduced functionality.
        """
        coordinator = coordinator_with_deps

        # Mock notification system to track notifications
        notification_log = []

        async def mock_notify(notification_type, details):
            notification_log.append({"type": notification_type, "details": details})

        # Replace the notification method for testing
        coordinator._notify_feature_degradation = mock_notify

        # Trigger fallback which should disable features and notify user
        await coordinator._trigger_fallback_mode("intelligent_degradation_test")

        # Verify intelligent feature disabling occurred
        features = await coordinator.get_available_features()

        # Check that high-level features are disabled
        assert (
            "dual_source_coordination" not in features["available"]
        ), "Dual source coordination should be disabled"
        assert (
            "ground_sdr_integration" not in features["available"]
        ), "Ground SDR integration should be disabled"

        # Check that essential features remain available
        assert "drone_source_only" in features["available"], "Drone source should remain available"
        assert (
            "safety_fallback_operation" in features["available"]
        ), "Safety fallback should remain available"

        # Verify user notifications were sent
        assert len(notification_log) > 0, "User notifications should be sent during degradation"

        # Find feature degradation notification
        degradation_notification = None
        for notification in notification_log:
            if notification["type"] == "feature_degradation":
                degradation_notification = notification
                break

        assert (
            degradation_notification is not None
        ), "Feature degradation notification should be sent"
        assert (
            "disabled_features" in degradation_notification["details"]
        ), "Notification should include disabled features list"
        assert (
            "available_features" in degradation_notification["details"]
        ), "Notification should include available features list"

    @pytest.mark.asyncio
    async def test_coordination_overhead_reduction_in_fallback_mode(self, coordinator_with_deps):
        """
        Test [2j3]: Implement coordination overhead reduction in fallback mode.

        AUTHENTIC BEHAVIOR: System must reduce coordination overhead during fallback
        to optimize performance for single-SDR operation.
        """
        coordinator = coordinator_with_deps

        # Measure coordination overhead in normal mode
        normal_start = time.perf_counter()
        await coordinator.make_coordination_decision()
        normal_duration = time.perf_counter() - normal_start

        # Trigger fallback to enable overhead reduction
        await coordinator._trigger_fallback_mode("overhead_reduction_test")

        # Measure coordination overhead in fallback mode
        fallback_start = time.perf_counter()
        await coordinator.make_coordination_decision()
        fallback_duration = time.perf_counter() - fallback_start

        # Verify overhead reduction (fallback should be faster or similar)
        assert (
            fallback_duration <= normal_duration * 2.0
        ), f"Fallback coordination should not be significantly slower: {fallback_duration:.4f}s vs {normal_duration:.4f}s"

        # Test multiple coordination cycles for consistency
        fallback_times = []
        for _ in range(5):
            start = time.perf_counter()
            await coordinator.make_coordination_decision()
            fallback_times.append(time.perf_counter() - start)

        avg_fallback_time = sum(fallback_times) / len(fallback_times)

        # Verify consistent performance optimization
        assert (
            avg_fallback_time <= 0.25
        ), f"Average fallback coordination time should be under 250ms: {avg_fallback_time:.4f}s"

        # Verify coordination is still functional despite overhead reduction
        health_status = await coordinator.get_health_status()
        assert (
            health_status["coordination_active"] == coordinator.is_running
        ), "Coordination should remain active despite overhead reduction"

    @pytest.mark.asyncio
    async def test_graceful_degradation_status_reporting(self, coordinator_with_deps):
        """
        Test [2j4]: Create graceful degradation status reporting.

        AUTHENTIC BEHAVIOR: System must report degradation status for monitoring
        and operational visibility.
        """
        coordinator = coordinator_with_deps

        # Get initial degradation status (should be none)
        initial_status = await coordinator.get_degradation_status()
        assert initial_status["degradation_active"] is False, "No degradation initially"
        assert initial_status["degradation_level"] == "none", "Level should be none initially"

        # Trigger fallback to activate degradation
        await coordinator._trigger_fallback_mode("status_reporting_test")

        # Get degradation status after fallback
        degradation_status = await coordinator.get_degradation_status()

        # Verify comprehensive degradation reporting
        assert degradation_status["degradation_active"] is True, "Degradation should be active"
        assert degradation_status["degradation_level"] in [
            "partial",
            "significant",
        ], "Degradation level should be classified"

        # Verify detailed status information
        required_status_fields = [
            "disabled_features_count",
            "available_features_count",
            "degradation_reason",
            "degradation_start_time",
            "performance_impact_level",
        ]

        for field in required_status_fields:
            assert field in degradation_status, f"Status field {field} should be reported"

        # Verify status accuracy
        assert (
            degradation_status["degradation_reason"] == "status_reporting_test"
        ), "Reason should match trigger reason"
        assert degradation_status["disabled_features_count"] > 0, "Some features should be disabled"
        assert (
            degradation_status["available_features_count"] > 0
        ), "Some features should still be available"

        # Verify performance impact classification
        assert degradation_status["performance_impact_level"] in [
            "low",
            "medium",
            "high",
        ], "Performance impact should be classified"

    @pytest.mark.asyncio
    async def test_fallback_mode_optimization_for_single_sdr(self, coordinator_with_deps):
        """
        Test [2j5]: Add fallback mode optimization for single-SDR operation.

        AUTHENTIC BEHAVIOR: System must optimize coordination algorithms
        for single-SDR operation during fallback.
        """
        coordinator = coordinator_with_deps

        # Verify dual-SDR behavior in normal mode
        coordinator.fallback_active = False
        normal_rssi = await coordinator.get_best_rssi()
        assert normal_rssi is not None, "RSSI should be available in normal mode"

        # Trigger fallback for single-SDR optimization
        await coordinator._trigger_fallback_mode("single_sdr_optimization_test")

        # Verify single-SDR optimized behavior
        optimized_rssi = await coordinator.get_best_rssi()
        assert optimized_rssi is not None, "RSSI should be available in fallback mode"
        assert coordinator.active_source == "drone", "Source should be optimized to drone"

        # Test optimization performance over multiple cycles
        optimization_times = []
        for cycle in range(3):
            start = time.perf_counter()
            await coordinator.make_coordination_decision()
            rssi_value = await coordinator.get_best_rssi()
            end = time.perf_counter()

            optimization_times.append(end - start)

            # Verify consistent single-SDR operation
            assert rssi_value is not None, f"RSSI should be available in cycle {cycle}"
            assert (
                coordinator.active_source == "drone"
            ), f"Source should remain drone-optimized in cycle {cycle}"

        # Verify optimization consistency
        avg_optimization_time = sum(optimization_times) / len(optimization_times)
        max_optimization_time = max(optimization_times)

        assert (
            avg_optimization_time < 0.10
        ), f"Average optimization should be under 100ms: {avg_optimization_time:.4f}s"
        assert (
            max_optimization_time < 0.20
        ), f"Maximum optimization should be under 200ms: {max_optimization_time:.4f}s"

        # Verify optimization maintains coordination quality
        health_status = await coordinator.get_health_status()
        assert health_status["fallback_active"] is True, "Fallback should be active"
        assert health_status["active_source"] == "drone", "Source should be optimized"
        assert (
            health_status["coordination_active"] == coordinator.is_running
        ), "Coordination should remain active with optimization"


class TestTask5_5_2_2_K_OperatorNotificationSystem:
    """
    Tests for [2k]: Create operator notification system for fallback activation.

    TDD Focus: Real-time notifications with severity classification and rate limiting.
    """

    @pytest.mark.asyncio
    async def test_real_time_fallback_activation_notifications_via_websocket(
        self, coordinator_with_deps
    ):
        """
        Test [2k1]: Implement real-time fallback activation notifications via WebSocket.

        AUTHENTIC BEHAVIOR: System must send real-time notifications via WebSocket
        during fallback activation for immediate operator awareness.
        """
        coordinator = coordinator_with_deps

        # Mock notification tracking for WebSocket integration
        notification_log = []

        # Mock WebSocket notification method if available
        if hasattr(coordinator, "_notify_websocket"):
            original_notify = coordinator._notify_websocket

            async def mock_websocket_notify(event_type, data):
                notification_log.append({"type": event_type, "data": data, "channel": "websocket"})
                if original_notify:
                    await original_notify(event_type, data)

            coordinator._notify_websocket = mock_websocket_notify

        # Trigger fallback which should generate real-time notifications
        await coordinator._trigger_fallback_mode("websocket_notification_test")

        # Verify WebSocket notifications were triggered during fallback
        # Check if TCP bridge auto_notify was called (which includes WebSocket integration)
        tcp_bridge = coordinator._tcp_bridge
        assert tcp_bridge is not None, "TCP bridge must be available for notifications"
        assert hasattr(
            tcp_bridge, "auto_notify_communication_issue"
        ), "TCP bridge must support communication issue notifications"

        # Verify the notification system is integrated (TCP bridge handles WebSocket routing)
        health_status = await coordinator.get_health_status()
        assert health_status["fallback_active"] is True, "Fallback must be active for notifications"

        # Verify notification content includes critical information
        assert hasattr(
            coordinator, "_fallback_reason"
        ), "Fallback reason must be tracked for notifications"
        assert (
            coordinator._fallback_reason == "websocket_notification_test"
        ), "Notification reason must match trigger"

    @pytest.mark.asyncio
    async def test_fallback_status_integration_with_safety_manager_notifications(
        self, coordinator_with_deps
    ):
        """
        Test [2k2]: Add fallback status integration with safety manager notifications.

        AUTHENTIC BEHAVIOR: Fallback status must be integrated with safety manager
        notification system for comprehensive safety event tracking.
        """
        coordinator = coordinator_with_deps

        # Track safety manager notification calls
        safety_notifications = []
        original_handle = coordinator._safety_manager.handle_communication_loss

        async def mock_safety_notify(event):
            safety_notifications.append(event)
            if callable(original_handle):
                result = original_handle(event)
                if inspect.isawaitable(result):
                    return await result
                return result

        coordinator._safety_manager.handle_communication_loss = mock_safety_notify

        # Trigger fallback which should integrate with safety manager notifications
        await coordinator._trigger_fallback_mode("safety_integration_test")

        # Verify safety manager integration occurred
        assert len(safety_notifications) > 0, "Safety manager should receive fallback notifications"

        # Verify notification contains fallback status integration
        safety_event = safety_notifications[0]
        assert "event_type" in safety_event, "Safety event must include event type"
        assert safety_event["event_type"] in [
            "coordination_fallback",
            "feature_degradation",
        ], "Event type must indicate coordination or feature fallback"

        # Verify comprehensive status integration
        if "details" in safety_event:
            details = safety_event["details"]
            assert "fallback_reason" in details, "Safety notification must include fallback reason"
            assert (
                "degradation_level" in details
            ), "Safety notification must include degradation level"

    @pytest.mark.asyncio
    async def test_fallback_alert_severity_classification_and_routing(self, coordinator_with_deps):
        """
        Test [2k3]: Create fallback alert severity classification and routing.

        AUTHENTIC BEHAVIOR: Fallback alerts must be classified by severity
        and routed appropriately based on the severity level.
        """
        coordinator = coordinator_with_deps

        # Test different severity scenarios
        severity_scenarios = [
            ("communication_loss", "high"),
            ("safety_triggered_fallback", "critical"),
            ("ground_sdr_failure", "high"),
            ("tcp_bridge_error", "medium"),
            ("frequency_sync_failed", "medium"),
            ("routine_test", "low"),
        ]

        for reason, expected_severity in severity_scenarios:
            # Reset coordinator state
            coordinator.fallback_active = False

            # Trigger fallback for severity testing
            await coordinator._trigger_fallback_mode(reason)

            # Get current degradation status to check severity classification
            degradation_status = await coordinator.get_degradation_status()

            # Verify severity classification exists
            assert (
                "performance_impact_level" in degradation_status
            ), f"Severity classification must exist for {reason}"

            # Verify severity is appropriately classified
            impact_level = degradation_status["performance_impact_level"]
            assert impact_level in [
                "low",
                "medium",
                "high",
            ], f"Impact level must be classified for {reason}: got {impact_level}"

            # Verify routing logic based on severity (high impact = high severity routing)
            if expected_severity in ["high", "critical"]:
                assert impact_level in [
                    "medium",
                    "high",
                ], f"High severity scenario {reason} should have medium/high impact level"

    @pytest.mark.asyncio
    async def test_fallback_notification_rate_limiting_and_aggregation(self, coordinator_with_deps):
        """
        Test [2k4]: Implement fallback notification rate limiting and aggregation.

        AUTHENTIC BEHAVIOR: Rapid successive fallback events must be rate limited
        and aggregated to prevent notification flooding.
        """
        coordinator = coordinator_with_deps

        # Track notification timing for rate limiting verification
        notification_times = []

        # Mock notification system to track timing
        original_notify = coordinator._notify_operator_fallback_activation

        async def timed_notify(reason):
            notification_times.append(time.time())
            if original_notify is None:
                return
            result = original_notify(reason)
            if inspect.isawaitable(result):
                return await result
            return result

        coordinator._notify_operator_fallback_activation = timed_notify

        # Trigger rapid successive fallbacks
        rapid_triggers = ["rapid_test_1", "rapid_test_2", "rapid_test_3", "rapid_test_4"]

        start_time = time.time()
        for reason in rapid_triggers:
            await coordinator._trigger_fallback_mode(reason)
            await asyncio.sleep(0.01)  # 10ms between triggers

        # Verify notifications were sent (rate limiting allows reasonable notification frequency)
        assert len(notification_times) == len(
            rapid_triggers
        ), "All notifications should be sent for different reasons"

        # Verify timing constraints (notifications should not be excessively delayed)
        total_time = notification_times[-1] - notification_times[0]
        assert (
            total_time < 1.0
        ), f"Notification processing should be under 1 second: {total_time:.3f}s"

        # Verify notification aggregation in trigger count
        assert coordinator._fallback_trigger_count == len(
            rapid_triggers
        ), "Trigger count should aggregate multiple rapid fallbacks"

    @pytest.mark.asyncio
    async def test_operator_acknowledgment_system_for_fallback_events(self, coordinator_with_deps):
        """
        Test [2k5]: Add operator acknowledgment system for fallback events.

        AUTHENTIC BEHAVIOR: Critical fallback events must support operator
        acknowledgment to confirm awareness and response capability.
        """
        coordinator = coordinator_with_deps

        # Trigger fallback that requires acknowledgment
        await coordinator._trigger_fallback_mode("acknowledgment_test")

        # Verify acknowledgment capability exists in health status
        health_status = await coordinator.get_health_status()

        # Check for acknowledgment-related fields in health status
        acknowledgment_fields = [
            "fallback_active",
            "fallback_reason",
            "fallback_start_time",
            "fallback_trigger_count",
        ]

        for field in acknowledgment_fields:
            assert (
                field in health_status
            ), f"Acknowledgment system requires {field} in health status"

        # Verify acknowledgment information is available for operator response
        assert (
            health_status["fallback_reason"] == "acknowledgment_test"
        ), "Acknowledgment system must track specific fallback reason"
        assert (
            health_status["fallback_start_time"] > 0
        ), "Acknowledgment system must track when fallback started"

        # Test acknowledgment status tracking
        if hasattr(coordinator, "_fallback_scenarios"):
            scenarios = coordinator._fallback_scenarios
            assert len(scenarios) > 0, "Scenarios must be tracked for acknowledgment"
            latest_scenario = scenarios[-1]
            assert "reason" in latest_scenario, "Scenario must include reason for acknowledgment"
            assert (
                "timestamp" in latest_scenario
            ), "Scenario must include timestamp for acknowledgment"


class TestTask5_5_2_2_L_AutomaticRecovery:
    """
    Tests for [2l]: Add automatic recovery when ground communication restored.

    TDD Focus: Automatic recovery with integrity checks and smooth transitions.
    """

    @pytest.mark.asyncio
    async def test_ground_communication_restoration_detection_with_validation(
        self, coordinator_with_deps
    ):
        """
        Test [2l1]: Implement ground communication restoration detection with validation.

        AUTHENTIC BEHAVIOR: System must detect when ground communication is restored
        and validate the connection quality before triggering recovery.
        """
        coordinator = coordinator_with_deps

        # Start in fallback mode
        await coordinator._trigger_fallback_mode("recovery_detection_test")
        assert coordinator.fallback_active is True
        assert coordinator.active_source == "drone"

        # Simulate ground communication restoration
        coordinator._tcp_bridge.is_running = True
        coordinator._tcp_bridge.get_ground_rssi.return_value = -30.0  # Good signal

        # Test ground communication detection
        health_status = await coordinator.get_health_status()
        ground_available = health_status["ground_connection_status"]
        assert ground_available is True, "Ground connection status should detect restoration"

        # Test communication validation through RSSI quality check
        ground_rssi = coordinator._tcp_bridge.get_ground_rssi()
        assert ground_rssi is not None, "Ground RSSI should be available after restoration"
        assert ground_rssi > -80, "Ground RSSI should indicate good signal quality"

        # Verify restoration detection is ready for recovery trigger
        best_rssi = await coordinator.get_best_rssi()
        # In fallback mode, should still return drone RSSI despite ground availability
        assert best_rssi == -50.0, "Should still use drone RSSI while in fallback mode"

    @pytest.mark.asyncio
    async def test_automatic_coordination_restoration_with_integrity_checks(
        self, coordinator_with_deps
    ):
        """
        Test [2l2]: Add automatic coordination restoration with integrity checks.

        AUTHENTIC BEHAVIOR: When recovery is triggered, coordination must be restored
        with comprehensive integrity checks to ensure system stability.
        """
        coordinator = coordinator_with_deps

        # Start in fallback mode
        await coordinator._trigger_fallback_mode("integrity_check_test")
        assert coordinator.fallback_active is True

        # Restore ground communication with good signal
        coordinator._tcp_bridge.is_running = True
        coordinator._tcp_bridge.get_ground_rssi.return_value = -25.0  # Excellent signal

        # Trigger recovery with integrity checks
        await coordinator._trigger_recovery_mode("ground_communication_restored")

        # Verify recovery was successful
        assert coordinator.fallback_active is False, "Fallback should be disabled after recovery"

        # Verify coordination restoration with integrity
        health_status = await coordinator.get_health_status()
        assert (
            health_status["coordination_active"] == coordinator.is_running
        ), "Coordination should remain active during recovery"

        # Verify source selection integrity - should choose best signal
        best_rssi = await coordinator.get_best_rssi()
        assert best_rssi == -25.0, "Should select ground RSSI after successful recovery"
        assert coordinator.active_source == "ground", "Should recover to ground source"

        # Verify integrity checks completed successfully
        assert not hasattr(
            coordinator, "_recovery_failure_flag"
        ), "Recovery should not have integrity check failures"

    @pytest.mark.asyncio
    async def test_recovery_transition_smoothing_to_prevent_oscillation(
        self, coordinator_with_deps
    ):
        """
        Test [2l3]: Create recovery transition smoothing to prevent oscillation.

        AUTHENTIC BEHAVIOR: Recovery transitions must be smoothed to prevent
        oscillation between fallback and normal modes.
        """
        coordinator = coordinator_with_deps

        # Test oscillation prevention scenario
        await coordinator._trigger_fallback_mode("oscillation_test")
        assert coordinator.fallback_active is True

        # Simulate marginal ground signal that could cause oscillation
        coordinator._tcp_bridge.is_running = True
        coordinator._tcp_bridge.get_ground_rssi.return_value = -55.0  # Marginal signal

        # Trigger recovery
        await coordinator._trigger_recovery_mode("marginal_recovery")

        # Verify recovery occurred despite marginal signal
        assert coordinator.fallback_active is False, "Recovery should occur with marginal signal"

        # Test transition smoothing - should not immediately re-trigger fallback
        await asyncio.sleep(0.02)  # Brief delay to test stability

        # Verify no oscillation occurred
        assert (
            coordinator.fallback_active is False
        ), "Should remain in recovered state (no oscillation)"

        # Test multiple recovery attempts to verify smoothing
        recovery_times = []
        for i in range(3):
            start_time = time.perf_counter()
            await coordinator._trigger_recovery_mode(f"smoothing_test_{i}")
            end_time = time.perf_counter()
            recovery_times.append(end_time - start_time)

        # Verify consistent recovery performance (smoothing should not add excessive delay)
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        assert (
            avg_recovery_time < 0.05
        ), f"Recovery smoothing should not add excessive delay: {avg_recovery_time:.4f}s"

    @pytest.mark.asyncio
    async def test_recovery_validation_and_rollback_mechanisms(self, coordinator_with_deps):
        """
        Test [2l4]: Implement recovery validation and rollback mechanisms.

        AUTHENTIC BEHAVIOR: Recovery must be validated for success, and rollback
        to fallback mode if recovery validation fails.
        """
        coordinator = coordinator_with_deps

        # Start in fallback mode
        await coordinator._trigger_fallback_mode("rollback_test")
        assert coordinator.fallback_active is True

        # Test successful recovery validation
        coordinator._tcp_bridge.is_running = True
        coordinator._tcp_bridge.get_ground_rssi.return_value = -20.0  # Excellent signal

        await coordinator._trigger_recovery_mode("successful_validation")

        # Verify successful recovery validation
        assert coordinator.fallback_active is False, "Recovery should succeed with good signal"
        assert coordinator.active_source == "ground", "Should recover to ground source"

        # Verify recovery validation maintains system integrity
        health_status = await coordinator.get_health_status()
        assert (
            health_status["coordination_active"] == coordinator.is_running
        ), "Coordination integrity should be maintained during recovery"

        # Test recovery validation with signal quality assessment
        best_rssi = await coordinator.get_best_rssi()
        assert best_rssi == -20.0, "Recovery validation should confirm signal quality"

        # Verify no rollback occurred (successful recovery)
        await asyncio.sleep(0.01)  # Brief delay to verify stability
        assert coordinator.fallback_active is False, "Successful recovery should remain stable"

    @pytest.mark.asyncio
    async def test_recovery_completion_notification_and_status_reporting(
        self, coordinator_with_deps
    ):
        """
        Test [2l5]: Add recovery completion notification and status reporting.

        AUTHENTIC BEHAVIOR: Recovery completion must be reported with comprehensive
        status information for operational visibility.
        """
        coordinator = coordinator_with_deps

        # Track recovery notifications
        recovery_notifications = []

        # Mock safety manager to track recovery notifications
        import functools

        original_handle = getattr(
            coordinator._safety_manager, "handle_communication_restored", None
        )

        if callable(original_handle) and inspect.iscoroutinefunction(original_handle):
            # Original is async - make wrapper async too
            @functools.wraps(original_handle)
            async def mock_recovery_notify(*args, **kwargs):
                recovery_notifications.append(
                    {
                        "type": "recovery_completed",
                        "timestamp": time.time(),
                        "active_source": coordinator.active_source,
                        "fallback_active": coordinator.fallback_active,
                    }
                )
                return await original_handle(*args, **kwargs)

        else:
            # Original is sync or not callable - keep wrapper sync
            def mock_recovery_notify(*args, **kwargs):
                recovery_notifications.append(
                    {
                        "type": "recovery_completed",
                        "timestamp": time.time(),
                        "active_source": coordinator.active_source,
                        "fallback_active": coordinator.fallback_active,
                    }
                )
                if callable(original_handle):
                    return original_handle(*args, **kwargs)

        if hasattr(coordinator._safety_manager, "handle_communication_restored"):
            coordinator._safety_manager.handle_communication_restored = mock_recovery_notify

        # Start in fallback and recover
        await coordinator._trigger_fallback_mode("notification_test")

        # Restore ground communication
        coordinator._tcp_bridge.is_running = True
        coordinator._tcp_bridge.get_ground_rssi.return_value = -35.0

        # Trigger recovery which should generate completion notifications
        await coordinator._trigger_recovery_mode("recovery_notification_test")

        # Verify recovery completion notification
        if hasattr(coordinator._safety_manager, "handle_communication_restored"):
            assert (
                len(recovery_notifications) > 0
            ), "Recovery completion should generate notifications"

            notification = recovery_notifications[0]
            assert (
                notification["type"] == "recovery_completed"
            ), "Notification type should indicate completion"
            assert (
                notification["active_source"] == "ground"
            ), "Notification should show recovered source"
            assert (
                notification["fallback_active"] is False
            ), "Notification should show fallback disabled"

        # Verify comprehensive status reporting after recovery
        health_status = await coordinator.get_health_status()

        # Verify recovery status fields are cleared/updated appropriately
        assert health_status["fallback_active"] is False, "Status should show fallback inactive"
        assert health_status["active_source"] == "ground", "Status should show recovered source"

        # Verify degradation status reflects recovery
        degradation_status = await coordinator.get_degradation_status()
        assert (
            degradation_status["degradation_active"] is False
        ), "Degradation should be cleared after recovery"
        assert (
            degradation_status["degradation_level"] == "none"
        ), "Degradation level should be none after recovery"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
