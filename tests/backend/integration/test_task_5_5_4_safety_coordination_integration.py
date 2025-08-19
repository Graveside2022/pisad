"""
TASK-5.5.4-INTEGRATION-TESTING: Comprehensive Safety Integration Testing
SUBTASK-5.5.4.1 [4a] - Create integration tests for safety manager + dual SDR coordinator

Test suite for validating safety manager integration with dual SDR coordination system.
Ensures all safety mechanisms work correctly during coordination operations.

PRD References:
- PRD-AC5.5.5: Comprehensive safety integration validation
- PRD-FR16: Emergency stop <500ms response time
- PRD-NFR12: Deterministic timing for safety-critical functions
"""

import asyncio
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


class TestSafetyCoordinationIntegration:
    """Integration tests for safety manager and dual SDR coordinator."""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager instance."""
        manager = SafetyManager()
        await manager.start_monitoring()
        yield manager
        # SafetyManager doesn't have explicit stop method

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager instance."""
        return SafetyAuthorityManager()

    @pytest.fixture
    def dual_sdr_coordinator(self, safety_authority_manager):
        """Create dual SDR coordinator with safety integration."""
        return DualSDRCoordinator(safety_authority=safety_authority_manager)

    @pytest.mark.asyncio
    async def test_safety_manager_coordination_startup(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.1] - Safety manager and coordination system startup integration.

        Validates that safety manager integrates properly during coordinator startup.
        """
        # Start coordination with safety integration
        await dual_sdr_coordinator.start()

        # Verify safety authority is accessible
        assert dual_sdr_coordinator.safety_authority is not None
        assert dual_sdr_coordinator.safety_authority == safety_authority_manager

        # Verify safety manager is operational (monitoring task should be active)
        assert safety_manager.monitoring_task is not None
        assert not safety_manager.monitoring_task.done()

        # Verify coordination status is available
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status
        assert "healthy" in coordination_status

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_emergency_stop_during_coordination(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.2] - Emergency stop functionality during active coordination.

        Validates emergency stop works correctly during coordination operations.
        """
        await dual_sdr_coordinator.start()

        # Trigger emergency stop through safety manager
        start_time = time.time()

        emergency_decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            authority_level=SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
            reason="Test emergency stop during coordination",
            timestamp=time.time(),
        )

        # Emergency stop should be approved and executed
        approved = await safety_authority_manager.validate_safety_decision(emergency_decision)
        assert approved.approved is True

        # Apply emergency stop to safety manager
        emergency_result = safety_manager.trigger_emergency_stop()

        # Verify timing requirements (< 500ms per PRD-FR16)
        response_time = emergency_result["response_time_ms"]
        assert (
            response_time < 500
        ), f"Emergency stop took {response_time:.2f}ms, exceeds 500ms requirement"

        # Verify emergency stop was successful
        assert emergency_result["success"] is True

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_safety_decision_coordination_override(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.3] - Safety decisions override coordination operations.

        Validates safety authority can override coordination decisions.
        """
        await dual_sdr_coordinator.start()

        # Create coordination override decision
        override_decision = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
            authority_level=SafetyAuthorityLevel.FLIGHT_MODE_MONITOR,
            reason="Flight mode not GUIDED",
            timestamp=time.time(),
        )

        # Safety authority should approve override
        approved = await safety_authority_manager.validate_safety_decision(override_decision)
        assert approved.approved is True
        assert approved.authority_level == SafetyAuthorityLevel.FLIGHT_MODE_MONITOR

        # Trigger emergency safety override in coordinator
        override_result = await dual_sdr_coordinator.trigger_emergency_safety_override(
            "Flight mode not GUIDED - test override"
        )

        # Verify override was successful
        assert override_result["success"] is True
        assert "emergency_override" in override_result

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_safety_monitoring_during_coordination(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.4] - Safety monitoring continues during coordination operations.

        Validates safety monitoring is not degraded by coordination activities.
        """
        await dual_sdr_coordinator.start()

        # Simulate coordination operations
        for i in range(5):
            await asyncio.sleep(0.01)  # Small delay

        # Verify safety monitoring is active
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status

        # Verify coordination latency monitoring
        latency_status = safety_manager.get_coordination_latency_status()
        assert "coordination_latency_ms" in latency_status
        assert latency_status["within_threshold"] is True

        # Verify safety manager state is operational
        assert safety_manager.get_state() in ["IDLE", "OPERATIONAL"]

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordination_health_integration(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.5] - Coordination health monitoring integration with safety system.

        Validates safety system monitors coordination health and responds appropriately.
        """
        await dual_sdr_coordinator.start()

        # Get health status from coordinator
        health_status = await dual_sdr_coordinator.get_health_status()
        assert "status" in health_status

        # Verify safety system coordination status
        coordination_status = safety_manager.get_coordination_status()
        assert "healthy" in coordination_status

        # Verify battery health monitoring integration
        battery_health = safety_manager.get_coordination_battery_health()
        assert "ground" in battery_health
        assert "drone" in battery_health

        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_safety_and_coordination_operations(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.6] - Concurrent safety and coordination operations without interference.

        Validates safety operations don't interfere with coordination and vice versa.
        """
        await dual_sdr_coordinator.start()

        # Start concurrent operations
        coordination_task = asyncio.create_task(
            self._run_coordination_operations(dual_sdr_coordinator)
        )
        safety_task = asyncio.create_task(self._run_safety_operations(safety_manager))

        # Wait for both to complete
        coordination_results, safety_results = await asyncio.gather(
            coordination_task, safety_task, return_exceptions=True
        )

        # Verify no exceptions occurred
        assert not isinstance(
            coordination_results, Exception
        ), f"Coordination error: {coordination_results}"
        assert not isinstance(safety_results, Exception), f"Safety error: {safety_results}"

        # Verify both systems completed operations
        assert coordination_results["operations_completed"] > 0
        assert safety_results["operations_completed"] > 0

        await dual_sdr_coordinator.stop()

    async def _run_coordination_operations(self, coordinator):
        """Helper method for concurrent coordination operations."""
        operations_completed = 0

        for i in range(10):
            # Get status instead of non-existent methods
            await coordinator.get_health_status()
            operations_completed += 1
            await asyncio.sleep(0.005)  # 5ms between operations

        return {"operations_completed": operations_completed}

    async def _run_safety_operations(self, safety_manager):
        """Helper method for concurrent safety operations."""
        operations_completed = 0

        for i in range(20):
            # Use available safety manager methods
            safety_manager.get_coordination_status()
            safety_manager.get_state()
            operations_completed += 1
            await asyncio.sleep(0.002)  # 2ms between checks

        return {"operations_completed": operations_completed}

    # SUBTASK-5.5.4.2 Emergency Scenarios Implementation
    # [4h] Flight mode override during coordination operations
    @pytest.mark.asyncio
    async def test_flight_mode_override_during_coordination(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4h] - Flight mode override during coordination operations.

        Validates that flight mode changes immediately override coordination
        operations and disable homing regardless of coordination state.

        PRD-FR15: System shall immediately cease sending velocity commands
        when flight controller mode changes from GUIDED to any other mode.
        """
        # Create safety interlock system for authentic testing
        from src.backend.utils.safety import SafetyInterlockSystem

        safety_system = SafetyInterlockSystem()
        await safety_system.start_monitoring()

        await dual_sdr_coordinator.start()

        # Enable coordination
        safety_system.set_coordination_system(dual_sdr_coordinator, active=True)

        # Start in GUIDED mode with coordination active
        safety_system.update_flight_mode("GUIDED")
        await safety_system.enable_homing()

        # Verify homing is enabled
        safety_results = await safety_system.check_all_safety()
        assert safety_results["operator"] is True, "Homing should be enabled"
        assert safety_results["coordination_health"] is True, "Coordination should be active"

        # Test flight mode override - change to non-GUIDED mode
        start_time = time.time()
        safety_system.update_flight_mode("STABILIZE")

        # Check safety immediately after mode change
        safety_results = await safety_system.check_all_safety()

        # Verify immediate override (< 100ms per PRD-FR15)
        override_time = (time.time() - start_time) * 1000
        assert (
            override_time < 100
        ), f"Flight mode override took {override_time:.2f}ms, exceeds 100ms"

        # Verify homing is immediately disabled due to mode change
        assert safety_results["mode"] is False, "Mode check should fail for non-GUIDED mode"

        # Verify overall safety check fails (disabling homing)
        overall_safe = all(safety_results.values())
        assert overall_safe is False, "Overall safety should fail when mode is not GUIDED"

        # Test with other non-GUIDED modes
        for mode in ["LOITER", "RTL", "AUTO"]:
            # Re-enable homing in GUIDED mode
            safety_system.update_flight_mode("GUIDED")
            await safety_system.enable_homing()

            # Verify homing is enabled
            safety_results = await safety_system.check_all_safety()
            assert (
                safety_results["operator"] is True
            ), f"Homing should be enabled before {mode} test"

            # Override with new mode
            start_time = time.time()
            safety_system.update_flight_mode(mode)

            # Check safety after mode change
            safety_results = await safety_system.check_all_safety()
            override_time = (time.time() - start_time) * 1000

            # Verify override timing and effect
            assert (
                override_time < 100
            ), f"Mode {mode} override took {override_time:.2f}ms, exceeds 100ms"
            assert safety_results["mode"] is False, f"Mode check should fail for {mode}"

        await safety_system.stop_monitoring()
        await dual_sdr_coordinator.stop()

    # [4i] Geofence violation handling with dual SDR sources
    @pytest.mark.asyncio
    async def test_geofence_violation_dual_sdr_sources(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4i] - Geofence violation handling with dual SDR sources.

        Validates that geofence violations trigger safety responses
        regardless of which SDR source is active (ground or drone).

        PRD-FR8: Drone shall maintain flight within designated geofence
        boundaries with automatic enforcement.
        """
        # Create safety interlock system for authentic testing
        from src.backend.utils.safety import SafetyInterlockSystem

        safety_system = SafetyInterlockSystem()
        await safety_system.start_monitoring()

        await dual_sdr_coordinator.start()

        # Enable coordination and homing
        safety_system.set_coordination_system(dual_sdr_coordinator, active=True)
        safety_system.update_flight_mode("GUIDED")
        await safety_system.enable_homing()

        # Set geofence boundary (radius in meters from home position)
        safety_system.set_geofence(radius=5000.0, altitude=150.0)  # 5km radius, 150m altitude

        # Test position inside geofence (should be safe)
        safety_system.update_position(lat=37.7749, lon=-122.4194, alt=100.0)  # SF coordinates
        safety_results = await safety_system.check_all_safety()
        assert safety_results["geofence"] is True, "Position should be inside geofence"

        # Test geofence violation - position outside boundary
        start_time = time.time()
        safety_system.update_position(lat=37.8500, lon=-122.5500, alt=100.0)  # Outside boundary

        # Check geofence status
        safety_results = await safety_system.check_all_safety()
        response_time = (time.time() - start_time) * 1000

        # Verify immediate geofence response
        assert response_time < 500, f"Geofence response took {response_time:.2f}ms"
        assert safety_results["geofence"] is False, "Geofence violation should be detected"

        # Verify overall safety check fails (triggering RTL)
        overall_safe = all(safety_results.values())
        assert overall_safe is False, "Overall safety should fail on geofence violation"

        # Test geofence with altitude violation
        safety_system.update_position(lat=37.7749, lon=-122.4194, alt=200.0)  # Above altitude limit
        safety_results = await safety_system.check_all_safety()
        assert safety_results["geofence"] is False, "Altitude violation should be detected"

        # Test recovery - return to safe zone
        safety_system.update_position(lat=37.7749, lon=-122.4194, alt=100.0)  # Back inside
        safety_results = await safety_system.check_all_safety()
        assert safety_results["geofence"] is True, "Position should be safe after return"

        # Verify coordination system remains active during geofence operations
        assert safety_results["coordination_health"] is True, "Coordination should remain active"
        assert (
            safety_results["dual_source_signal"] is True
        ), "Dual source monitoring should continue"

        await safety_system.stop_monitoring()
        await dual_sdr_coordinator.stop()

    # [4j] Battery critical scenarios with coordination active
    @pytest.mark.asyncio
    async def test_battery_critical_coordination_active(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4j] - Battery critical scenarios with coordination active.

        Validates that battery critical conditions trigger proper safety
        responses while coordination system is active.

        PRD-FR10: System shall execute automatic return-to-launch (RTL)
        on low battery. Battery monitoring preserved with dual-SDR coordination.
        """
        # Create safety interlock system for authentic testing
        from src.backend.utils.safety import SafetyInterlockSystem

        safety_system = SafetyInterlockSystem()
        await safety_system.start_monitoring()

        await dual_sdr_coordinator.start()

        # Enable coordination and homing with good battery
        safety_system.set_coordination_system(dual_sdr_coordinator, active=True)
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(75.0)  # Good battery level (75%)
        await safety_system.enable_homing()

        # Verify homing is enabled with good battery
        safety_results = await safety_system.check_all_safety()
        assert safety_results["operator"] is True, "Homing should be enabled with good battery"
        assert safety_results["battery"] is True, "Battery check should pass with 75%"
        assert safety_results["coordination_health"] is True, "Coordination should be active"

        # Test low battery scenario (but not critical)
        safety_system.update_battery(25.0)  # Low but above 20% threshold
        safety_results = await safety_system.check_all_safety()
        assert safety_results["battery"] is True, "Battery check should still pass at 25%"

        # Homing should still be allowed with warning level
        overall_safe = all(safety_results.values())
        assert overall_safe is True, "System should be safe with 25% battery"

        # Test critical battery scenario
        start_time = time.time()
        safety_system.update_battery(15.0)  # Below 20% threshold - critical

        # Check safety immediately
        safety_results = await safety_system.check_all_safety()
        response_time = (time.time() - start_time) * 1000

        # Verify immediate safety response
        assert response_time < 500, f"Battery critical response took {response_time:.2f}ms"
        assert safety_results["battery"] is False, "Battery check should fail at 15%"

        # Verify overall safety check fails (triggering RTL)
        overall_safe = all(safety_results.values())
        assert overall_safe is False, "Overall safety should fail with critical battery"

        # Test very critical battery scenario
        safety_system.update_battery(5.0)  # Very low
        safety_results = await safety_system.check_all_safety()
        assert safety_results["battery"] is False, "Battery check should fail at 5%"

        # Test battery recovery
        safety_system.update_battery(80.0)  # Good battery restored
        safety_results = await safety_system.check_all_safety()
        assert safety_results["battery"] is True, "Battery check should pass after recovery"

        # Verify coordination system remains active during battery monitoring
        assert safety_results["coordination_health"] is True, "Coordination should remain active"
        assert (
            safety_results["dual_source_signal"] is True
        ), "Dual source monitoring should continue"

        await safety_system.stop_monitoring()
        await dual_sdr_coordinator.stop()

    # [4g] Enhanced emergency stop test with coordination system active
    @pytest.mark.asyncio
    async def test_enhanced_emergency_stop_coordination_system_active(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4g] - Enhanced emergency stop with coordination system active.

        Validates emergency stop functionality works correctly during active
        dual SDR coordination operations with all safety pathways tested.

        PRD-FR16: Emergency stop <500ms response time regardless of coordination state.
        """
        await dual_sdr_coordinator.start()

        # Enable full coordination with both sources active
        safety_manager.set_flight_mode("GUIDED")
        await safety_manager.enable_homing()

        # Start coordination operations
        await dual_sdr_coordinator.start_coordination()

        # Verify coordination and homing are fully active
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status
        assert "dual_source" in coordination_status

        # Test emergency stop during active coordination
        start_time = time.time()

        # Trigger emergency stop through safety authority
        emergency_decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            authority_level=SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
            reason="SUBTASK-5.5.4.2 [4g] emergency stop test during coordination",
            timestamp=time.time(),
        )

        # Emergency stop should be approved immediately
        approved = await safety_authority_manager.validate_safety_decision(emergency_decision)
        assert approved.approved is True

        # Execute emergency stop through safety manager
        emergency_result = safety_manager.trigger_emergency_stop()

        # Verify timing requirements with coordination overhead
        response_time = emergency_result["response_time_ms"]
        total_time = (time.time() - start_time) * 1000

        assert (
            response_time < 500
        ), f"Emergency stop took {response_time:.2f}ms, exceeds 500ms requirement"
        assert (
            total_time < 600
        ), f"Total emergency response took {total_time:.2f}ms, excessive with coordination"

        # Verify emergency stop was successful
        assert emergency_result["success"] is True
        assert emergency_result["coordination_stopped"] is True

        # Verify coordination system responded to emergency stop
        coordination_status = safety_manager.get_coordination_status()
        assert "emergency_stop_active" in coordination_status

        # Verify homing is immediately disabled
        safety_status = safety_manager.get_state()
        assert safety_status in [
            "EMERGENCY_STOP",
            "DISABLED",
        ], f"Emergency stop should disable homing: {safety_status}"

        # Verify dual coordinator emergency override
        coordinator_status = await dual_sdr_coordinator.get_emergency_status()
        assert coordinator_status["emergency_stop_active"] is True
        assert coordinator_status["coordination_halted"] is True

        # Test emergency stop during source switching
        await dual_sdr_coordinator.reset_emergency_state()
        await safety_manager.enable_homing()

        # Start source switching
        switching_task = asyncio.create_task(dual_sdr_coordinator.switch_source("ground", "drone"))

        # Trigger emergency stop during switching
        await asyncio.sleep(0.01)  # Let switching start
        emergency_result = safety_manager.trigger_emergency_stop()

        # Wait for switching to complete/abort
        await switching_task

        # Verify emergency stop worked during switching
        assert emergency_result["success"] is True
        assert emergency_result["source_switching_aborted"] is True

        await dual_sdr_coordinator.stop()
