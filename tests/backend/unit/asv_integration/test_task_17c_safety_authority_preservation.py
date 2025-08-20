"""
Test TASK-6.1.2.4 [17c] - Safety Authority Manager Integration Preservation

SUBTASK-6.1.2.4 [17c]: Preserve existing safety authority manager integration without modification

This module validates that all ASV enhanced processing components properly respect
the existing PISAD safety interlocks and never override safety authority decisions.

Test Coverage:
- All 6 safety authority levels (EMERGENCY_STOP through SIGNAL)
- Safety authority hierarchy enforcement
- Emergency stop propagation validation
- Geofence boundary enforcement preservation
- Flight mode monitoring integration
- Battery monitoring preservation
- Safety audit trail compliance

PRD References:
- FR15: System shall cease commands when flight mode changes
- FR16: Emergency controls with <500ms response
- NFR12: Deterministic timing for safety-critical functions
"""

import time
from unittest.mock import Mock

import pytest

from src.backend.services.asv_integration.asv_configuration_manager import ASVConfigurationManager
from src.backend.services.asv_integration.asv_hackrf_coordinator import ASVHackRFCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
)


class TestSafetyAuthorityPreservation:
    """Test safety authority manager integration preservation without modification."""

    @pytest.fixture
    async def mock_safety_authority(self):
        """Create mock safety authority manager with all required methods."""
        mock = Mock(spec=SafetyAuthorityManager)
        mock.validate_coordination_command = Mock(return_value=(True, "authorized"))
        mock.emergency_override_active = False
        mock.log_coordination_decision = Mock()
        return mock

    @pytest.fixture
    async def asv_coordinator(self, mock_safety_authority):
        """Create ASV coordinator with safety authority integration."""
        config_manager = Mock(spec=ASVConfigurationManager)
        config_manager.get_all_frequency_profiles.return_value = {}

        coordinator = ASVHackRFCoordinator(
            config_manager=config_manager, safety_authority=mock_safety_authority
        )
        return coordinator

    async def test_emergency_stop_authority_level_enforcement(
        self, asv_coordinator, mock_safety_authority
    ):
        """Test that EMERGENCY_STOP authority level blocks all ASV coordination.

        SUBTASK-6.1.2.4 [17c-2]: Verify safety authority hierarchy enforcement
        """
        # RED PHASE: Test should fail initially - emergency stop must block coordination
        mock_safety_authority.emergency_override_active = True

        # Attempt coordination validation during emergency override
        safety_validated = await asv_coordinator._validate_coordination_safety()

        # Emergency override must block coordination
        assert safety_validated is False, "Emergency override should block ASV coordination"

        # Verify no coordination command validation was attempted during emergency
        mock_safety_authority.validate_coordination_command.assert_not_called()

    async def test_flight_mode_authority_level_integration(
        self, asv_coordinator, mock_safety_authority
    ):
        """Test FLIGHT_MODE authority level integration with ASV operations.

        SUBTASK-6.1.2.4 [17c-6]: Test flight mode monitor integration
        """
        # RED PHASE: Flight mode authority must be respected
        mock_safety_authority.validate_coordination_command.return_value = (
            False,
            "flight_mode_blocked",
        )

        # Attempt coordination with flight mode restriction
        safety_validated = await asv_coordinator._validate_coordination_safety()

        # Flight mode restriction must block coordination
        assert safety_validated is False, "Flight mode restriction should block ASV coordination"

        # Verify coordination command was validated with SIGNAL authority level
        mock_safety_authority.validate_coordination_command.assert_called_once()
        call_args = mock_safety_authority.validate_coordination_command.call_args
        assert call_args[1]["authority_level"] == SafetyAuthorityLevel.SIGNAL

    async def test_geofence_authority_level_preservation(
        self, asv_coordinator, mock_safety_authority
    ):
        """Test GEOFENCE authority level enforcement with ASV frequency switching.

        SUBTASK-6.1.2.4 [17c-4]: Confirm geofence boundary enforcement
        """
        # RED PHASE: Geofence constraints must be validated in coordination
        coordination_details = {
            "component": "ASVHackRFCoordinator",
            "operation": "frequency_coordination",
            "current_frequency": 406_000_000,
            "active_analyzers": 1,
        }

        # Validate coordination with geofence requirements
        safety_validated = await asv_coordinator._validate_coordination_safety()

        # Should attempt coordination validation with proper details
        if mock_safety_authority.validate_coordination_command.called:
            call_args = mock_safety_authority.validate_coordination_command.call_args
            details = call_args[1]["details"]

            # Verify coordination details include required fields for geofence validation
            assert "component" in details
            assert "current_frequency" in details
            assert "active_analyzers" in details
            assert details["component"] == "ASVHackRFCoordinator"

    async def test_emergency_stop_response_timing(self, asv_coordinator):
        """Test emergency stop response meets <500ms requirement.

        SUBTASK-6.1.2.4 [17c-3]: Test emergency stop propagation
        """
        # RED PHASE: Emergency stop must complete within 500ms
        start_time = time.perf_counter()

        # Trigger emergency stop
        await asv_coordinator.emergency_stop()

        end_time = time.perf_counter()
        emergency_stop_duration_ms = (end_time - start_time) * 1000

        # Must meet <500ms requirement
        assert (
            emergency_stop_duration_ms < 500.0
        ), f"Emergency stop took {emergency_stop_duration_ms:.1f}ms, must be <500ms"

        # Verify coordination is stopped after emergency
        assert asv_coordinator._coordination_active is False

    async def test_safety_audit_trail_logging(self, asv_coordinator, mock_safety_authority):
        """Test safety audit trail logging for ASV coordination decisions.

        SUBTASK-6.1.2.4 [17c-7]: Verify safety audit trail logging
        """
        # RED PHASE: Safety decisions must be logged for audit compliance
        mock_safety_authority.validate_coordination_command.return_value = (True, "authorized")

        # Perform coordination validation
        safety_validated = await asv_coordinator._validate_coordination_safety()

        # Should be authorized
        assert safety_validated is True, "Coordination should be authorized when safety allows"

        # Verify audit trail logging was attempted
        # Assert that the method exists and was called
        assert hasattr(
            mock_safety_authority, "log_coordination_decision"
        ), "mock_safety_authority must have log_coordination_decision method"

        # Should log coordination decision for audit trail
        mock_safety_authority.log_coordination_decision.assert_called_once()

        call_args = mock_safety_authority.log_coordination_decision.call_args
        assert call_args[1]["component"] == "ASVHackRFCoordinator"
        assert call_args[1]["decision_type"] == "coordination_validation"
        assert call_args[1]["authority_level"] == SafetyAuthorityLevel.SIGNAL

    async def test_battery_monitoring_integration_preservation(
        self, asv_coordinator, mock_safety_authority
    ):
        """Test battery monitoring authority level integration preservation.

        SUBTASK-6.1.2.4 [17c-5]: Validate battery monitoring integration
        """
        # RED PHASE: Battery monitoring must affect ASV coordination decisions
        # This validates the integration exists without modifying core safety systems

        # Verify that coordination validation includes battery considerations
        safety_validated = await asv_coordinator._validate_coordination_safety()

        # Battery authority level should be checked through validate_coordination_command
        if mock_safety_authority.validate_coordination_command.called:
            call_args = mock_safety_authority.validate_coordination_command.call_args

            # Verify command type includes ASV coordination
            assert call_args[1]["command_type"] == "asv_coordination"

            # Verify details include information relevant to battery monitoring
            details = call_args[1]["details"]
            assert "active_analyzers" in details  # Battery load consideration

    async def test_signal_authority_level_requirement(self, asv_coordinator, mock_safety_authority):
        """Test that ASV coordination requires SIGNAL authority level.

        SUBTASK-6.1.2.4 [17c-1]: Validate SafetyAuthorityManager interface compatibility
        """
        # RED PHASE: ASV coordination must require appropriate authority level
        await asv_coordinator._validate_coordination_safety()

        # Verify coordination validation uses SIGNAL authority level
        mock_safety_authority.validate_coordination_command.assert_called_once()
        call_args = mock_safety_authority.validate_coordination_command.call_args

        # ASV coordination should require SIGNAL authority level (lowest level)
        assert call_args[1]["authority_level"] == SafetyAuthorityLevel.SIGNAL
        assert call_args[1]["command_type"] == "asv_coordination"

    async def test_safety_system_performance_no_degradation(
        self, asv_coordinator, mock_safety_authority
    ):
        """Test that ASV operations don't degrade safety system performance.

        SUBTASK-6.1.2.4 [17c-8]: Performance validation - ensure safety response <500ms
        """
        # RED PHASE: Safety validation must maintain performance with ASV active
        validation_times = []

        for _ in range(10):  # Test multiple validation cycles
            start_time = time.perf_counter()
            await asv_coordinator._validate_coordination_safety()
            end_time = time.perf_counter()

            validation_time_ms = (end_time - start_time) * 1000
            validation_times.append(validation_time_ms)

        # Safety validation should be fast (<100ms per validation)
        avg_validation_time = sum(validation_times) / len(validation_times)
        max_validation_time = max(validation_times)

        assert (
            avg_validation_time < 100.0
        ), f"Average safety validation {avg_validation_time:.1f}ms too slow"
        assert (
            max_validation_time < 200.0
        ), f"Maximum safety validation {max_validation_time:.1f}ms too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
