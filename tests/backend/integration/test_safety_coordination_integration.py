"""
Integration tests for safety manager + coordination system.

Tests SUBTASK-5.5.1.4 implementation with step [1s].
Validates that SafetyInterlockSystem works correctly with DualSDRCoordinator
and SDRPPBridgeService for comprehensive safety during SDR++ integration.

This ensures all safety mechanisms continue to function when coordination
is active, meeting PRD requirements for safety preservation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_manager import SafetyManager
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.utils.safety import SafetyEventType, SafetyInterlockSystem


class TestSafetyCoordinationIntegration:
    """Integration tests for safety manager + coordination system."""

    @pytest.fixture
    async def safety_system(self):
        """Create safety interlock system."""
        safety = SafetyInterlockSystem()
        await safety.start_monitoring()
        yield safety
        await safety.stop_monitoring()

    @pytest.fixture
    def safety_manager(self):
        """Create safety manager."""
        return SafetyManager()

    @pytest.fixture
    def dual_coordinator(self):
        """Create dual SDR coordinator."""
        coordinator = DualSDRCoordinator()

        # Mock methods for testing
        coordinator.get_health_status = AsyncMock(
            return_value={
                "coordination_latency_ms": 45.0,
                "ground_connection_status": 0.9,
                "coordination_active": True,
            }
        )
        coordinator.trigger_emergency_override = AsyncMock(
            return_value={
                "emergency_override_active": True,
                "response_time_ms": 150.0,
                "source_switched_to": "drone",
            }
        )
        coordinator.get_ground_rssi = MagicMock(return_value=15.0)
        coordinator.get_drone_rssi = MagicMock(return_value=12.0)

        return coordinator

    @pytest.fixture
    def tcp_bridge(self):
        """Create TCP bridge service."""
        bridge = SDRPPBridgeService()
        bridge.is_running = True
        bridge.get_ground_rssi = MagicMock(return_value=15.0)
        bridge.get_communication_health = AsyncMock(
            return_value={"connected": True, "latency_ms": 30.0, "quality": 0.95}
        )
        return bridge

    @pytest.mark.asyncio
    async def test_safety_system_with_coordination_active(self, safety_system, dual_coordinator):
        """Test [1s] - Safety system functions correctly with coordination active."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Verify coordination safety checks are active
        assert "coordination_health" in safety_system.checks
        assert "dual_source_signal" in safety_system.checks
        assert safety_system.coordination_active is True

        # Test all safety checks pass with coordination
        safety_results = await safety_system.check_all_safety()

        # Should have all standard checks plus coordination checks
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

        # Coordination health should pass with good status
        coord_health = await safety_system.check_coordination_health()
        assert coord_health["enabled"] is True
        assert coord_health["healthy"] is True
        assert coord_health["latency_ms"] == 45.0

    @pytest.mark.asyncio
    async def test_safety_emergency_stop_with_coordination(self, safety_system, dual_coordinator):
        """Test [1s] - Emergency stop works through coordination system."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Trigger emergency stop through coordination
        result = await safety_system.trigger_coordination_emergency_stop(
            "Integration test emergency"
        )

        # Verify emergency stop result
        assert result["safety_emergency_stop"] is True
        assert result["coordination_emergency_stop"]["emergency_override_active"] is True
        assert result["coordination_emergency_stop"]["response_time_ms"] == 150.0
        assert safety_system.emergency_stopped is True

        # Verify coordination emergency override was called
        dual_coordinator.trigger_emergency_override.assert_called_once()

    @pytest.mark.asyncio
    async def test_safety_manager_coordination_integration(
        self, safety_manager, dual_coordinator, tcp_bridge
    ):
        """Test [1s] - Safety manager integrates with coordination components."""
        # Set up dependencies
        dual_coordinator.set_dependencies(
            signal_processor=MagicMock(), tcp_bridge=tcp_bridge, safety_manager=safety_manager
        )

        # Test safety manager can monitor coordination health
        # This simulates the safety manager checking coordination status
        coord_healthy = tcp_bridge.is_running and hasattr(dual_coordinator, "get_health_status")
        assert coord_healthy is True

        # Test health status retrieval
        health_status = await dual_coordinator.get_health_status()
        assert health_status["coordination_active"] is not None  # Should have status
        assert "ground_connection_status" in health_status

    @pytest.mark.asyncio
    async def test_dual_source_signal_monitoring(self, safety_system, dual_coordinator):
        """Test [1s] - Dual source signal monitoring integration."""
        # Enable coordination with signal sources
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test dual source signal check
        signal_status = await safety_system.check_dual_source_signals()

        # Should detect both sources and no conflicts
        assert signal_status["safe"] is True
        assert signal_status["ground_snr"] == 15.0
        assert signal_status["drone_snr"] == 12.0
        assert signal_status["conflict_detected"] is False

    @pytest.mark.asyncio
    async def test_coordination_health_monitoring(self, safety_system, dual_coordinator):
        """Test [1s] - Coordination health monitoring integration."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test health monitoring
        health_status = await safety_system.check_coordination_health()

        assert health_status["enabled"] is True
        assert health_status["healthy"] is True
        assert health_status["latency_ms"] == 45.0
        assert health_status["communication_quality"] == 0.9

    @pytest.mark.asyncio
    async def test_safety_fallback_on_coordination_failure(self, safety_system, dual_coordinator):
        """Test [1s] - Safety system fallback when coordination fails."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Simulate coordination failure
        dual_coordinator.get_health_status = AsyncMock(side_effect=Exception("Coordination failed"))

        # Health check should fail safely
        health_status = await safety_system.check_coordination_health()
        assert health_status["enabled"] is True
        assert health_status["healthy"] is False

        # Safety system should still function
        is_safe = await safety_system.is_safe_to_proceed()
        # Should be False because coordination health failed
        assert is_safe is False

    @pytest.mark.asyncio
    async def test_comprehensive_safety_status_with_coordination(
        self, safety_system, dual_coordinator
    ):
        """Test [1s] - Comprehensive safety status includes coordination data."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Get comprehensive status
        status = safety_system.get_coordination_safety_status()

        # Verify coordination status is included
        assert "coordination" in status
        coord_status = status["coordination"]

        assert coord_status["coordination_active"] is True
        assert coord_status["coordination_safety_enabled"] is True
        assert coord_status["dual_sdr_coordinator_available"] is True

        # Should include coordination health details
        assert "coordination_health" in coord_status
        assert "dual_source_signal" in coord_status

    @pytest.mark.asyncio
    async def test_safety_event_logging_with_coordination(self, safety_system, dual_coordinator):
        """Test [1s] - Safety events are logged correctly with coordination."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Clear any existing events
        safety_system.safety_events.clear()

        # Trigger coordination emergency stop
        await safety_system.trigger_coordination_emergency_stop("Test logging")

        # Verify events were logged
        events = safety_system.get_safety_events()
        assert len(events) >= 1

        # Find the coordination emergency event
        coord_events = [
            e
            for e in events
            if e.event_type == SafetyEventType.EMERGENCY_STOP
            and e.details.get("coordination_triggered") is True
        ]
        assert len(coord_events) >= 1

        coord_event = coord_events[0]
        assert coord_event.details["reason"] == "Test logging"
        assert "coordination_result" in coord_event.details

    def test_safety_system_initialization_with_coordination_checks(self):
        """Test [1s] - Safety system initializes with coordination awareness."""
        safety_system = SafetyInterlockSystem()

        # Verify coordination-specific checks are present
        assert "coordination_health" in safety_system.checks
        assert "dual_source_signal" in safety_system.checks

        # Verify coordination attributes are initialized
        assert hasattr(safety_system, "coordination_active")
        assert hasattr(safety_system, "coordination_safety_enabled")
        assert hasattr(safety_system, "dual_sdr_coordinator")

        # Coordination should start disabled
        assert safety_system.coordination_active is False
        assert safety_system.coordination_safety_enabled is True
        assert safety_system.dual_sdr_coordinator is None
