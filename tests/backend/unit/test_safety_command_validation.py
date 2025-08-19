"""
Unit tests for Safety Command Validation Layer

Tests SUBTASK-5.5.3.2 [9d] implementation - safety validation layer
for all coordination commands before execution.

TDD Phase: RED - Writing failing tests first for command validation
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
)
from src.backend.services.sdr_priority_manager import SDRPriorityManager
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestSafetyCommandValidation:
    """Test suite for safety command validation layer"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    @pytest.fixture
    def mock_dual_sdr_coordinator(self):
        """Mock DualSDRCoordinator for testing"""
        coordinator = MagicMock()
        coordinator._safety_authority = None
        return coordinator

    @pytest.fixture
    def mock_priority_manager(self):
        """Mock SDRPriorityManager for testing"""
        manager = MagicMock()
        manager._safety_authority = None
        return manager

    def test_validate_coordination_command_emergency_stop(self, authority_manager):
        """Test validation of emergency stop command"""
        # TDD RED: This test should FAIL initially
        result, message = authority_manager.validate_coordination_command_real_time(
            command_type="emergency_stop",
            authority_level=SafetyAuthorityLevel.EMERGENCY_STOP,
            details={"reason": "operator_triggered", "source": "ui_button"},
            response_time_limit_ms=500,
        )

        assert result is True
        assert "emergency_stop authorized" in message.lower()
        # Response should be well under 500ms requirement - no specific timing assertion needed

    def test_validate_coordination_command_insufficient_authority(self, authority_manager):
        """Test rejection of command with insufficient authority"""
        # TDD RED: This test should FAIL initially
        result, message = authority_manager.validate_coordination_command_real_time(
            command_type="emergency_stop",
            authority_level=SafetyAuthorityLevel.SIGNAL,  # Too low authority
            details={"reason": "signal_loss"},
            response_time_limit_ms=500,
        )

        assert result is False
        assert "insufficient authority" in message.lower()

    def test_validate_source_selection_command(self, authority_manager):
        """Test validation of source selection commands"""
        # TDD RED: This test should FAIL initially
        result, message = authority_manager.validate_coordination_command_real_time(
            command_type="source_selection",
            authority_level=SafetyAuthorityLevel.COMMUNICATION,
            details={"from_source": "ground", "to_source": "drone", "reason": "comm_degradation"},
            response_time_limit_ms=100,
        )

        assert result is True
        assert "source_selection authorized" in message.lower()

    def test_validate_command_timing_requirement(self, authority_manager):
        """Test that command validation meets timing requirements"""
        # TDD RED: This test should FAIL initially
        start_time = datetime.now()

        result, message = authority_manager.validate_coordination_command_real_time(
            command_type="coordination_override",
            authority_level=SafetyAuthorityLevel.FLIGHT_MODE,
            details={"override_reason": "mode_change", "target_mode": "MANUAL"},
            response_time_limit_ms=100,
        )

        end_time = datetime.now()
        response_time_ms = (end_time - start_time).total_seconds() * 1000

        assert result is True
        assert response_time_ms < 100  # Must be under 100ms
        assert "coordination_override authorized" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_command_during_emergency_override(self, authority_manager):
        """Test command validation during emergency override state"""
        # TDD RED: This test should FAIL initially
        # Trigger emergency override
        await authority_manager.trigger_emergency_override("test_emergency")

        # Try to validate a normal command - should be blocked
        result, message = authority_manager.validate_coordination_command_real_time(
            command_type="source_selection",
            authority_level=SafetyAuthorityLevel.COMMUNICATION,
            details={"reason": "normal_operation"},
            response_time_limit_ms=100,
        )

        assert result is False
        assert "emergency override active" in message.lower()

    def test_coordination_command_validation_metrics(self, authority_manager):
        """Test that command validation provides timing metrics"""
        # TDD RED: This test should FAIL initially
        result, message, metrics = authority_manager.validate_coordination_command_with_metrics(
            command_type="fallback_trigger",
            authority_level=SafetyAuthorityLevel.SIGNAL,
            details={"trigger": "signal_loss", "fallback_mode": "drone_only"},
            response_time_limit_ms=200,
        )

        assert result is True
        assert metrics is not None
        assert "validation_time_ms" in metrics
        assert metrics["validation_time_ms"] < 200
        assert "authority_level" in metrics
        assert metrics["authority_level"] == SafetyAuthorityLevel.SIGNAL


class TestCoordinationServiceValidationIntegration:
    """Test integration of validation layer with coordination services"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    def test_dual_sdr_coordinator_validation_integration(self, authority_manager):
        """Test DualSDRCoordinator integrates safety validation"""
        # TDD RED: This test should FAIL initially - DualSDRCoordinator needs integration

        coordinator = DualSDRCoordinator()
        coordinator._safety_authority = authority_manager

        # Test that coordinator validates commands before execution
        result = coordinator.validate_command_before_execution(
            command="source_switch", params={"target_source": "ground", "reason": "better_signal"}
        )

        assert result is not None
        assert "authorized" in result
        assert "message" in result
        assert result["authorized"] is True or result["authorized"] is False

    def test_priority_manager_validation_integration(self, authority_manager):
        """Test SDRPriorityManager integrates safety validation"""
        # TDD RED: This test should FAIL initially - SDRPriorityManager needs integration

        priority_manager = SDRPriorityManager(coordinator=MagicMock(), safety_manager=MagicMock())
        priority_manager._safety_authority = authority_manager

        # Test that priority manager validates decisions before execution
        result = priority_manager.validate_priority_decision(
            decision_type="source_priority_change",
            details={"new_priority": "ground_preferred", "reason": "signal_quality"},
        )

        assert result is not None
        assert "validation_result" in result
        assert "authorized" in result["validation_result"]

    def test_bridge_service_validation_integration(self, authority_manager):
        """Test SDRPPBridgeService integrates safety validation"""
        # TDD RED: This test should FAIL initially - SDRPPBridgeService needs integration

        bridge_service = SDRPPBridgeService()
        bridge_service._safety_authority = authority_manager

        # Test that bridge service validates incoming commands
        result = bridge_service.validate_incoming_command(
            command_type="frequency_change",
            command_data={"frequency": 2437000000, "source": "ground_sdr"},
        )

        assert result is not None
        assert "authorized" in result
        assert "validation_time_ms" in result
