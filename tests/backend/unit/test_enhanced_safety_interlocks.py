"""
Test enhanced SafetyInterlockSystem with SDR++ coordination awareness.

Tests SUBTASK-5.5.1.2 implementation with steps [1g] through [1l].
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.utils.safety import (
    CoordinationHealthCheck,
    DualSourceSignalCheck,
    SafetyEventType,
    SafetyInterlockSystem,
    SafetyTrigger,
)


class TestEnhancedSafetyInterlocks:
    """Test enhanced safety interlock system with coordination awareness."""

    @pytest.fixture
    def safety_system(self):
        """Create enhanced safety system."""
        return SafetyInterlockSystem()

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock DualSDRCoordinator."""
        coordinator = AsyncMock()
        coordinator.get_health_status = AsyncMock(
            return_value={
                "coordination_latency_ms": 50.0,
                "ground_connection_status": 0.8,
                "coordination_active": True,
            }
        )
        coordinator.get_ground_rssi = MagicMock(return_value=10.0)  # Above 6dB threshold
        coordinator.get_drone_rssi = MagicMock(return_value=8.0)  # Above 6dB threshold
        coordinator.trigger_emergency_override = AsyncMock(
            return_value={"emergency_override_active": True, "response_time_ms": 250.0}
        )
        return coordinator

    def test_enhanced_initialization(self, safety_system):
        """Test [1g] - SafetyInterlockSystem includes coordination checks."""
        # Verify new coordination checks are included
        assert "coordination_health" in safety_system.checks
        assert "dual_source_signal" in safety_system.checks

        # Verify coordination checks are correct types
        coord_check = safety_system.checks["coordination_health"]
        assert isinstance(coord_check, CoordinationHealthCheck)

        signal_check = safety_system.checks["dual_source_signal"]
        assert isinstance(signal_check, DualSourceSignalCheck)

        # Verify coordination attributes
        assert hasattr(safety_system, "dual_sdr_coordinator")
        assert hasattr(safety_system, "coordination_active")
        assert hasattr(safety_system, "coordination_safety_enabled")
        assert safety_system.coordination_safety_enabled is True

    def test_set_coordination_system(self, safety_system, mock_coordinator):
        """Test [1g] - Set coordination system for safety monitoring."""
        # Test setting coordination system
        safety_system.set_coordination_system(mock_coordinator, active=True)

        assert safety_system.dual_sdr_coordinator == mock_coordinator
        assert safety_system.coordination_active is True

        # Verify coordination checks are updated
        coord_check = safety_system.checks["coordination_health"]
        assert coord_check.coordination_active is True
        assert coord_check.dual_sdr_coordinator == mock_coordinator

        signal_check = safety_system.checks["dual_source_signal"]
        assert signal_check.dual_sdr_coordinator == mock_coordinator

    @pytest.mark.asyncio
    async def test_coordination_health_check(self, safety_system, mock_coordinator):
        """Test [1i] - Coordination health monitoring with safety events."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        # Test healthy coordination
        health_status = await safety_system.check_coordination_health()
        assert health_status["enabled"] is True
        assert health_status["healthy"] is True
        assert health_status["latency_ms"] == 50.0
        assert health_status["communication_quality"] == 0.8

    @pytest.mark.asyncio
    async def test_coordination_health_degraded(self, safety_system, mock_coordinator):
        """Test coordination health degradation triggers safety events."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        # Mock degraded health
        mock_coordinator.get_health_status.return_value = {
            "coordination_latency_ms": 150.0,  # Exceeds 100ms threshold
            "ground_connection_status": 0.3,  # Below 50% quality
            "coordination_active": True,
        }

        health_status = await safety_system.check_coordination_health()
        assert health_status["healthy"] is False
        assert health_status["latency_ms"] == 150.0
        assert health_status["communication_quality"] == 0.3

        # Verify safety event was logged
        coord_events = [
            e
            for e in safety_system.safety_events
            if e.event_type == SafetyEventType.COORDINATION_HEALTH_DEGRADED
        ]
        assert len(coord_events) > 0

    @pytest.mark.asyncio
    async def test_dual_source_signal_check(self, safety_system, mock_coordinator):
        """Test [1h] - Dual source signal quality assessment."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        # Test normal operation
        signal_status = await safety_system.check_dual_source_signals()
        assert signal_status["safe"] is True
        assert signal_status["ground_snr"] == 10.0
        assert signal_status["drone_snr"] == 8.0
        assert signal_status["conflict_detected"] is False

    @pytest.mark.asyncio
    async def test_dual_source_conflict_detection(self, safety_system, mock_coordinator):
        """Test dual source conflict detection and logging."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        # Mock conflicting signals (>10dB difference)
        mock_coordinator.get_ground_rssi.return_value = 20.0  # Strong signal
        mock_coordinator.get_drone_rssi.return_value = 8.0  # Weaker signal (12dB diff)

        signal_status = await safety_system.check_dual_source_signals()
        assert signal_status["safe"] is True  # Still safe, but conflict detected
        assert signal_status["conflict_detected"] is True

        # Verify conflict event was logged
        conflict_events = [
            e
            for e in safety_system.safety_events
            if e.event_type == SafetyEventType.DUAL_SOURCE_CONFLICT
        ]
        assert len(conflict_events) > 0
        assert conflict_events[0].details["difference"] == 12.0

    @pytest.mark.asyncio
    async def test_dual_source_both_sources_fail(self, safety_system, mock_coordinator):
        """Test when both sources fail threshold."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        # Mock both sources below threshold
        mock_coordinator.get_ground_rssi.return_value = -80.0  # Below -6dB threshold
        mock_coordinator.get_drone_rssi.return_value = -85.0  # Below threshold

        signal_status = await safety_system.check_dual_source_signals()
        assert signal_status["safe"] is False
        assert "Both sources below threshold" in signal_status["failure_reason"]

    @pytest.mark.asyncio
    async def test_coordination_emergency_stop(self, safety_system, mock_coordinator):
        """Test [1j] - Emergency stop through coordination system."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        result = await safety_system.trigger_coordination_emergency_stop("Test emergency")

        # Verify standard emergency stop was triggered
        assert safety_system.emergency_stopped is True
        assert result["safety_emergency_stop"] is True

        # Verify coordination emergency override was triggered
        mock_coordinator.trigger_emergency_override.assert_called_once()
        assert result["coordination_emergency_stop"]["emergency_override_active"] is True

        # Verify emergency event was logged
        emergency_events = [
            e
            for e in safety_system.safety_events
            if e.event_type == SafetyEventType.EMERGENCY_STOP
            and e.details.get("coordination_triggered")
        ]
        assert len(emergency_events) > 0

    def test_coordination_safety_status(self, safety_system, mock_coordinator):
        """Test [1k] - Comprehensive coordination safety status."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        status = safety_system.get_coordination_safety_status()

        # Verify base safety status is included
        assert "emergency_stopped" in status
        assert "checks" in status

        # Verify coordination status is included
        assert "coordination" in status
        coord_status = status["coordination"]
        assert coord_status["coordination_active"] is True
        assert coord_status["coordination_safety_enabled"] is True
        assert coord_status["dual_sdr_coordinator_available"] is True

        # Verify coordination check details
        assert "coordination_health" in coord_status
        assert "dual_source_signal" in coord_status

    @pytest.mark.asyncio
    async def test_latency_monitoring(self, safety_system):
        """Test [1l] - Coordination latency monitoring and events."""
        # Test acceptable latency
        result = await safety_system.coordination_latency_check(50.0)
        assert result is True

        # Test excessive latency
        result = await safety_system.coordination_latency_check(150.0)
        assert result is False

        # Verify latency violation event was logged
        latency_events = [
            e
            for e in safety_system.safety_events
            if e.event_type == SafetyEventType.COORDINATION_LATENCY_EXCEEDED
        ]
        assert len(latency_events) > 0
        assert latency_events[0].details["measured_latency_ms"] == 150.0

    def test_enable_disable_coordination_safety(self, safety_system):
        """Test enabling/disabling coordination safety monitoring."""
        # Test disabling
        safety_system.enable_coordination_safety(False)
        assert safety_system.coordination_safety_enabled is False

        # Test enabling
        safety_system.enable_coordination_safety(True)
        assert safety_system.coordination_safety_enabled is True

    @pytest.mark.asyncio
    async def test_coordination_safety_disabled(self, safety_system, mock_coordinator):
        """Test coordination health check when safety is disabled."""
        safety_system.set_coordination_system(mock_coordinator, active=True)
        safety_system.enable_coordination_safety(False)

        health_status = await safety_system.check_coordination_health()
        assert health_status["enabled"] is False
        assert health_status["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_coordination_inactive(self, safety_system):
        """Test coordination health check when coordination is inactive."""
        safety_system.coordination_active = False

        health_status = await safety_system.check_coordination_health()
        assert health_status["enabled"] is False
        assert health_status["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_emergency_stop_coordination_failure(self, safety_system, mock_coordinator):
        """Test emergency stop when coordination system fails."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        # Mock coordination emergency override failure
        mock_coordinator.trigger_emergency_override.side_effect = Exception("Coordination failed")

        result = await safety_system.trigger_coordination_emergency_stop("Test failure")

        # Verify safety emergency stop still worked
        assert result["safety_emergency_stop"] is True
        assert safety_system.emergency_stopped is True

        # Verify coordination failure was recorded
        assert "error" in result["coordination_emergency_stop"]
        assert "Coordination failed" in result["coordination_emergency_stop"]["error"]

    @pytest.mark.asyncio
    async def test_comprehensive_safety_checks_with_coordination(
        self, safety_system, mock_coordinator
    ):
        """Test that all safety checks work together with coordination."""
        safety_system.set_coordination_system(mock_coordinator, active=True)

        # Run all safety checks
        results = await safety_system.check_all_safety()

        # Verify all checks are included
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
            assert check_name in results

        # Verify coordination checks are working
        assert results["coordination_health"] is True  # Should be healthy with mock
        assert results["dual_source_signal"] is True  # Should be safe with mock

    def test_coordination_trigger_mapping(self, safety_system):
        """Test that coordination triggers are properly mapped."""
        # Test coordination health trigger
        trigger = safety_system._get_trigger_for_check("coordination_health")
        assert trigger == SafetyTrigger.COORDINATION_FAILURE

        # Test dual source signal trigger
        trigger = safety_system._get_trigger_for_check("dual_source_signal")
        assert trigger == SafetyTrigger.SOURCE_CONFLICT

    @pytest.mark.asyncio
    async def test_coordination_health_check_no_coordinator(self, safety_system):
        """Test coordination health check when no coordinator is set."""
        # Enable coordination but don't set coordinator
        safety_system.coordination_active = True
        safety_system.dual_sdr_coordinator = None

        coord_check = safety_system.checks["coordination_health"]
        coord_check.coordination_active = True
        coord_check.dual_sdr_coordinator = None

        # Should be safe when no coordinator (drone-only mode)
        is_safe = await coord_check.check()
        assert is_safe is True

    @pytest.mark.asyncio
    async def test_signal_check_no_coordinator(self, safety_system):
        """Test dual source signal check when no coordinator is set."""
        signal_check = safety_system.checks["dual_source_signal"]
        signal_check.dual_sdr_coordinator = None

        # Should still work with default values
        is_safe = await signal_check.check()
        # Both sources will be -100dB (below threshold), so should fail
        assert is_safe is False
        assert "Both sources below threshold" in signal_check.failure_reason
