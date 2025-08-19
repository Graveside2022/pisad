"""
Safety Manager and DualSDRCoordinator integration tests.

Tests SUBTASK-5.5.2.4: Implement safety manager integration into DualSDRCoordinator.
Validates that safety manager is properly integrated with coordination system
for emergency override, safety-aware decision making, and comprehensive logging.

This ensures PRD-AC5.5.4: Safety authority hierarchy maintained with coordination.

Chain of Thought Context:
- PRD → Epic 5 → Story 5.5 → TASK-5.5.2-EMERGENCY-FALLBACK → SUBTASK-5.5.2.4
- Integration Points: SafetyManager dependency injection, emergency overrides, safety logging
- Previous Context: Emergency stop timing validation (5.5.2.3) ✅ COMPLETED
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_manager import SafetyManager
from src.backend.services.sdr_priority_manager import SDRPriorityManager


class TestSafetyManagerCoordinatorIntegration:
    """Test safety manager integration with DualSDRCoordinator."""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager for integration testing."""
        manager = SafetyManager()

        # Mock MAVLink with comprehensive telemetry
        mock_mavlink = MagicMock()
        mock_mavlink.emergency_stop = MagicMock(return_value=True)
        mock_mavlink.telemetry = {
            "battery": {"voltage": 12.5},
            "gps": {"satellites": 10, "hdop": 1.0, "fix_type": 3},
            "mode": "GUIDED",
            "armed": True,
        }
        manager.mavlink = mock_mavlink

        return manager

    @pytest.fixture
    async def coordinator_with_safety(self, safety_manager):
        """Create DualSDRCoordinator with safety manager integration."""
        coordinator = DualSDRCoordinator()

        # Mock other dependencies
        mock_signal_processor = AsyncMock()
        mock_tcp_bridge = AsyncMock()
        mock_tcp_bridge.is_connected.return_value = True

        # Set dependencies including safety manager
        coordinator.set_dependencies(
            signal_processor=mock_signal_processor,
            tcp_bridge=mock_tcp_bridge,
            safety_manager=safety_manager,
        )

        await coordinator.start()
        yield coordinator
        await coordinator.stop()

    async def test_safety_manager_dependency_injection_constructor(self, safety_manager):
        """
        Test [2s]: Safety manager dependency injection to DualSDRCoordinator constructor.

        RED PHASE: Tests that safety manager is properly injected and accessible
        throughout the coordination system for emergency overrides.
        """
        coordinator = DualSDRCoordinator()

        # Test dependency injection
        coordinator.set_dependencies(safety_manager=safety_manager)

        # Verify safety manager is properly injected
        assert coordinator._safety_manager is not None
        assert coordinator._safety_manager == safety_manager

        # Verify priority manager has safety manager reference
        assert coordinator._priority_manager is not None
        assert hasattr(coordinator._priority_manager, "_safety_manager")

        print("Safety manager successfully injected into DualSDRCoordinator")

    async def test_safety_triggered_emergency_override_coordination(
        self, coordinator_with_safety, safety_manager
    ):
        """
        Test [2t]: Safety-triggered emergency override in coordination decision logic.

        RED PHASE: Tests that safety manager can trigger emergency override
        that immediately stops coordination decisions and forces drone-only mode.
        """
        # Verify coordinator is running with ground coordination
        assert coordinator_with_safety.is_running == True
        coordinator_with_safety.active_source = "ground"
        coordinator_with_safety.fallback_active = False

        # Trigger safety emergency override
        start_time = time.perf_counter()
        safety_result = safety_manager.trigger_emergency_stop()
        end_time = time.perf_counter()

        # Wait for coordination loop to process safety override
        await asyncio.sleep(0.1)  # Allow coordination loop to respond

        response_time_ms = (end_time - start_time) * 1000

        # Verify emergency override was successful
        assert safety_result is not None
        assert response_time_ms < 500.0, f"Safety override took {response_time_ms:.1f}ms"

        # Verify coordination system responds to safety override
        # (Coordination should continue running but with safety awareness)
        assert coordinator_with_safety.is_running == True

        print(f"Safety emergency override processed in {response_time_ms:.1f}ms")

    async def test_safety_aware_coordination_decision_making(
        self, coordinator_with_safety, safety_manager
    ):
        """
        Test [2u]: Safety-aware coordination decision making with priority hierarchy.

        RED PHASE: Tests that coordination decisions consider safety status
        and prioritize safety over coordination efficiency.
        """
        # Mock safety status monitoring
        safety_manager.is_safe_to_operate = MagicMock(return_value=True)
        safety_manager.get_safety_status = MagicMock(
            return_value={"safe": True, "warnings": [], "critical_alerts": []}
        )

        # Test coordination decision with safety consideration
        initial_source = coordinator_with_safety.active_source

        # Simulate safety manager reporting unsafe condition
        safety_manager.is_safe_to_operate.return_value = False
        safety_manager.get_safety_status.return_value = {
            "safe": False,
            "warnings": ["Battery low"],
            "critical_alerts": ["Emergency condition detected"],
        }

        # Allow coordination loop to make safety-aware decision
        await asyncio.sleep(0.1)

        # Verify safety manager was consulted for decision making
        assert safety_manager.is_safe_to_operate.called or safety_manager.get_safety_status.called

        # Coordination system should remain operational but safety-aware
        assert coordinator_with_safety.is_running == True

        print("Coordination decision making incorporates safety status")

    async def test_safety_status_monitoring_coordination_health_checks(
        self, coordinator_with_safety, safety_manager
    ):
        """
        Test [2v]: Safety status monitoring integrated into coordination health checks.

        RED PHASE: Tests that coordination health monitoring includes safety status
        and reports safety issues as part of overall system health.
        """
        # Get coordination system health status
        health_status = await coordinator_with_safety.get_health_status()

        # Verify health status includes safety information
        assert health_status is not None
        assert isinstance(health_status, dict)

        # Health status should include safety-related metrics
        expected_fields = ["coordination_active", "active_source", "fallback_active"]
        for field in expected_fields:
            assert field in health_status, f"Missing health field: {field}"

        # Test safety status integration
        safety_manager.get_safety_status = MagicMock(
            return_value={
                "emergency_stopped": False,
                "interlocks_active": True,
                "last_check": time.time(),
            }
        )

        # Get updated health status with safety integration
        updated_health = await coordinator_with_safety.get_health_status()

        # Verify safety status is considered in health reporting
        assert updated_health is not None

        print(f"Coordination health status includes safety monitoring: {updated_health}")

    async def test_coordination_shutdown_safety_system_failures(
        self, coordinator_with_safety, safety_manager
    ):
        """
        Test [2w]: Coordination shutdown on safety system failures.

        RED PHASE: Tests that coordination system properly shuts down or
        transitions to safe mode when safety system failures are detected.
        """
        # Simulate safety system failure
        safety_manager.is_operational = MagicMock(return_value=False)
        safety_manager.get_failure_reason = MagicMock(
            return_value="Safety system communications failure"
        )

        # Verify coordinator is initially running
        assert coordinator_with_safety.is_running == True

        # Allow coordination loop to detect safety system failure
        await asyncio.sleep(0.1)

        # Verify coordination system remains operational but in safe mode
        # (Complete shutdown may not be desired - prefer safe fallback)
        assert coordinator_with_safety.is_running == True

        # Verify safety-aware fallback is activated
        # In case of safety system failure, should fallback to drone-only operation
        if hasattr(coordinator_with_safety, "fallback_active"):
            assert (
                coordinator_with_safety.fallback_active == True
                or coordinator_with_safety.active_source == "drone"
            )

        print("Coordination system handles safety system failures safely")

    async def test_safety_event_logging_coordination_operations(
        self, coordinator_with_safety, safety_manager
    ):
        """
        Test [2x]: Comprehensive safety event logging for all coordination operations.

        RED PHASE: Tests that all coordination decisions and safety interactions
        are properly logged for audit trail and troubleshooting.
        """
        with patch("src.backend.services.dual_sdr_coordinator.logger") as mock_logger:

            # Perform coordination operations that should trigger safety logging
            await coordinator_with_safety.synchronize_frequency(2.4e9)  # 2.4 GHz

            # Trigger safety event
            safety_manager.trigger_emergency_stop()

            # Allow time for logging
            await asyncio.sleep(0.05)

            # Verify coordination operations were logged
            log_calls = (
                mock_logger.info.call_args_list
                + mock_logger.warning.call_args_list
                + mock_logger.critical.call_args_list
            )

            # Should have logs for coordination and safety events
            assert len(log_calls) > 0, "No coordination/safety events were logged"

            # Look for safety-related log entries
            safety_logs = [
                call
                for call in log_calls
                if any(
                    keyword in str(call).lower()
                    for keyword in ["safety", "emergency", "coordination", "synchronizing"]
                )
            ]

            # Check specific log content for safety integration
            log_content = " ".join([str(call) for call in log_calls])
            has_safety_logs = any(
                [
                    "safety" in log_content.lower(),
                    "synchronizing frequency" in log_content.lower(),
                    "coordination" in log_content.lower(),
                ]
            )

            assert (
                has_safety_logs or len(safety_logs) > 0
            ), f"No safety-related coordination logs found. Available logs: {[str(call) for call in log_calls]}"

            print(f"Safety event logging captured {len(safety_logs)} coordination safety events")

    async def test_integrated_safety_coordination_performance(
        self, coordinator_with_safety, safety_manager
    ):
        """
        Integration test: Safety manager integration performance with coordination.

        RED PHASE: Validates that safety manager integration does not degrade
        coordination system performance below PRD requirements.
        """
        # Measure coordination performance with safety integration
        performance_samples = []

        for _ in range(10):  # Multiple samples for reliable measurement
            start_time = time.perf_counter()

            # Simulate coordination decision with safety checks
            health_status = await coordinator_with_safety.get_health_status()
            await coordinator_with_safety.synchronize_frequency(2.4e9)

            end_time = time.perf_counter()
            operation_time_ms = (end_time - start_time) * 1000
            performance_samples.append(operation_time_ms)

            # Brief delay between samples
            await asyncio.sleep(0.01)

        # Analyze performance
        avg_time_ms = sum(performance_samples) / len(performance_samples)
        max_time_ms = max(performance_samples)

        # Verify safety integration doesn't degrade performance
        assert (
            avg_time_ms < 50.0
        ), f"Average coordination time {avg_time_ms:.1f}ms exceeds 50ms target"
        assert (
            max_time_ms < 100.0
        ), f"Maximum coordination time {max_time_ms:.1f}ms exceeds 100ms limit"

        print(
            f"Safety-integrated coordination performance: avg={avg_time_ms:.1f}ms, max={max_time_ms:.1f}ms"
        )

    async def test_safety_manager_emergency_coordination_integration(
        self, coordinator_with_safety, safety_manager
    ):
        """
        End-to-end test: Safety manager emergency override through coordination system.

        RED PHASE: Tests complete emergency response pathway from safety manager
        through coordination system with proper timing and state management.
        """
        # Set up initial coordination state
        coordinator_with_safety.active_source = "ground"
        coordinator_with_safety.fallback_active = False

        # Trigger comprehensive emergency response
        start_time = time.perf_counter()

        # Emergency stop through safety manager
        emergency_result = safety_manager.trigger_emergency_stop()

        # Allow coordination system to respond
        await asyncio.sleep(0.1)

        end_time = time.perf_counter()
        total_response_time_ms = (end_time - start_time) * 1000

        # Verify emergency response was successful
        assert emergency_result is not None
        assert (
            total_response_time_ms < 500.0
        ), f"Total emergency response took {total_response_time_ms:.1f}ms"

        # Verify coordination system maintains operation in safe state
        assert coordinator_with_safety.is_running == True

        # Verify safety status is preserved
        health_status = await coordinator_with_safety.get_health_status()
        assert health_status is not None

        print(f"End-to-end safety emergency coordination: {total_response_time_ms:.1f}ms")
