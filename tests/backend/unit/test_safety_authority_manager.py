"""
Unit tests for SafetyAuthorityManager

Tests SUBTASK-5.5.3.2 [9a] implementation with comprehensive
validation of safety authority hierarchy enforcement.

Validates that all safety decisions follow proper authority levels
and emergency response pathways per PRD requirements.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.backend.services.safety_authority_manager import (
    SafetyAuthority,
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
    SafetyDecision,
    SafetyDecisionType,
)


class TestSafetyAuthorityManager:
    """Unit tests for safety authority manager"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    def test_authority_hierarchy_initialization(self, authority_manager):
        """Test [9a] - Safety authority hierarchy is properly initialized"""
        # Verify all 6 authority levels are present
        assert len(authority_manager.authorities) == 6

        # Verify each authority level exists
        expected_levels = [
            SafetyAuthorityLevel.EMERGENCY_STOP,
            SafetyAuthorityLevel.FLIGHT_MODE,
            SafetyAuthorityLevel.GEOFENCE,
            SafetyAuthorityLevel.BATTERY,
            SafetyAuthorityLevel.COMMUNICATION,
            SafetyAuthorityLevel.SIGNAL,
        ]

        for level in expected_levels:
            assert level in authority_manager.authorities
            authority = authority_manager.authorities[level]
            assert isinstance(authority, SafetyAuthority)
            assert authority.active is True
            assert authority.level == level

    def test_authority_response_times(self, authority_manager):
        """Test [9a] - Authority response times meet PRD requirements"""
        # Verify response times per hierarchy
        response_times = {
            SafetyAuthorityLevel.EMERGENCY_STOP: 500,  # <500ms
            SafetyAuthorityLevel.FLIGHT_MODE: 100,  # <100ms
            SafetyAuthorityLevel.GEOFENCE: 1000,  # <1s
            SafetyAuthorityLevel.BATTERY: 5000,  # <5s
            SafetyAuthorityLevel.COMMUNICATION: 10000,  # <10s
            SafetyAuthorityLevel.SIGNAL: 10000,  # <10s
        }

        for level, expected_time in response_times.items():
            authority = authority_manager.authorities[level]
            assert authority.response_time_ms == expected_time

    def test_authority_integration_points(self, authority_manager):
        """Test [9a] - Authority integration points are documented"""
        expected_integration_points = {
            SafetyAuthorityLevel.EMERGENCY_STOP: "All coordination components",
            SafetyAuthorityLevel.FLIGHT_MODE: "DualSDRCoordinator decision making",
            SafetyAuthorityLevel.GEOFENCE: "Priority Manager source selection",
            SafetyAuthorityLevel.BATTERY: "Coordination health monitoring",
            SafetyAuthorityLevel.COMMUNICATION: "SDRPPBridge health monitoring",
            SafetyAuthorityLevel.SIGNAL: "Dual source signal validation",
        }

        for level, expected_point in expected_integration_points.items():
            authority = authority_manager.authorities[level]
            assert authority.integration_point == expected_point

    @pytest.mark.asyncio
    async def test_emergency_stop_decision_validation(self, authority_manager):
        """Test [9a] - Emergency stop decisions are properly validated"""
        # Create emergency stop decision
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
            details={"reason": "Test emergency stop"},
        )

        # Validate decision
        approved, reason, approving_authority = await authority_manager.validate_safety_decision(
            decision
        )

        # Verify approval
        assert approved is True
        assert approving_authority == SafetyAuthorityLevel.EMERGENCY_STOP
        assert decision.approved is True
        assert decision.response_time_ms is not None
        assert decision.response_time_ms <= 500  # Must meet <500ms requirement

    @pytest.mark.asyncio
    async def test_insufficient_authority_rejection(self, authority_manager):
        """Test [9a] - Insufficient authority requests are rejected"""
        # Try to request emergency stop with insufficient authority
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            requesting_authority=SafetyAuthorityLevel.SIGNAL,  # Level 6 - insufficient
            details={"reason": "Test insufficient authority"},
        )

        # Validate decision
        approved, reason, approving_authority = await authority_manager.validate_safety_decision(
            decision
        )

        # Verify rejection
        assert approved is False
        assert approving_authority is None
        assert "Insufficient authority" in reason
        assert decision.approved is False

    @pytest.mark.asyncio
    async def test_source_selection_authority_validation(self, authority_manager):
        """Test [9a] - Source selection requires appropriate authority level"""
        # Communication level can request source selection
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.SOURCE_SELECTION,
            requesting_authority=SafetyAuthorityLevel.COMMUNICATION,
            details={"target_source": "drone_only", "reason": "Ground communication lost"},
        )

        approved, reason, approving_authority = await authority_manager.validate_safety_decision(
            decision
        )

        assert approved is True
        assert approving_authority == SafetyAuthorityLevel.COMMUNICATION
        assert decision.response_time_ms <= 10000  # <10s requirement

    @pytest.mark.asyncio
    async def test_flight_mode_authority_validation(self, authority_manager):
        """Test [9a] - Flight mode decisions require level 2 authority"""
        # Flight mode level can request coordination override
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
            requesting_authority=SafetyAuthorityLevel.FLIGHT_MODE,
            details={"mode": "MANUAL", "reason": "Pilot override"},
        )

        approved, reason, approving_authority = await authority_manager.validate_safety_decision(
            decision
        )

        assert approved is True
        assert approving_authority == SafetyAuthorityLevel.FLIGHT_MODE
        assert decision.response_time_ms <= 100  # <100ms requirement

    @pytest.mark.asyncio
    async def test_emergency_override_bypass(self, authority_manager):
        """Test [9a] - Emergency override bypasses all authority checks"""
        # Trigger emergency override
        result = await authority_manager.trigger_emergency_override("Test emergency")

        assert result["emergency_override_active"] is True
        assert result["response_time_ms"] <= 500
        assert authority_manager.emergency_override_active is True

        # Any decision should now be auto-approved
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.SYSTEM_SHUTDOWN,
            requesting_authority=SafetyAuthorityLevel.SIGNAL,  # Normally insufficient
            details={"reason": "System shutdown during emergency"},
        )

        approved, reason, approving_authority = await authority_manager.validate_safety_decision(
            decision
        )

        assert approved is True
        assert approving_authority == SafetyAuthorityLevel.EMERGENCY_STOP
        assert "Emergency override active" in reason

    def test_coordination_command_validation(self, authority_manager):
        """Test [9a] - Coordination commands are validated against hierarchy"""
        # Flight mode command with sufficient authority
        valid, reason = authority_manager.validate_coordination_command(
            "mode_change", SafetyAuthorityLevel.FLIGHT_MODE, {"target_mode": "GUIDED"}
        )
        assert valid is True

        # Flight mode command with insufficient authority
        valid, reason = authority_manager.validate_coordination_command(
            "mode_change", SafetyAuthorityLevel.BATTERY, {"target_mode": "GUIDED"}
        )
        assert valid is False
        assert "Insufficient authority" in reason

    def test_emergency_override_blocks_commands(self, authority_manager):
        """Test [9a] - Emergency override blocks all coordination commands"""
        # Activate emergency override
        authority_manager.emergency_override_active = True

        # Any command should be blocked
        valid, reason = authority_manager.validate_coordination_command(
            "source_switch", SafetyAuthorityLevel.EMERGENCY_STOP, {}
        )

        assert valid is False
        assert "Emergency override active" in reason

    def test_authority_deactivation(self, authority_manager):
        """Test [9a] - Authority levels can be deactivated for maintenance"""
        # Deactivate communication authority
        authority_manager.deactivate_authority(
            SafetyAuthorityLevel.COMMUNICATION, "Maintenance mode"
        )

        # Verify deactivation
        assert authority_manager.authorities[SafetyAuthorityLevel.COMMUNICATION].active is False

        # Reactivate
        authority_manager.reactivate_authority(SafetyAuthorityLevel.COMMUNICATION)
        assert authority_manager.authorities[SafetyAuthorityLevel.COMMUNICATION].active is True

    @pytest.mark.asyncio
    async def test_inactive_authority_rejection(self, authority_manager):
        """Test [9a] - Inactive authorities cannot approve decisions"""
        # Deactivate flight mode authority
        authority_manager.deactivate_authority(SafetyAuthorityLevel.FLIGHT_MODE, "Testing")

        # Try to request coordination override
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
            requesting_authority=SafetyAuthorityLevel.FLIGHT_MODE,
            details={"reason": "Test with inactive authority"},
        )

        approved, reason, approving_authority = await authority_manager.validate_safety_decision(
            decision
        )

        assert approved is False
        assert "inactive" in reason.lower()

        # Reactivate for cleanup
        authority_manager.reactivate_authority(SafetyAuthorityLevel.FLIGHT_MODE)

    def test_authority_status_reporting(self, authority_manager):
        """Test [9a] - Authority status can be monitored"""
        status = authority_manager.get_authority_status()

        assert "emergency_override_active" in status
        assert "authorities" in status
        assert "recent_decisions" in status

        # Verify all 6 authority levels in status
        assert len(status["authorities"]) == 6

        for level_key, auth_status in status["authorities"].items():
            assert "name" in auth_status
            assert "level" in auth_status
            assert "active" in auth_status
            assert "response_time_ms" in auth_status
            assert "integration_point" in auth_status

    @pytest.mark.asyncio
    async def test_decision_audit_trail(self, authority_manager):
        """Test [9a] - Safety decisions are logged for audit trail"""
        # Make several decisions
        decisions = [
            SafetyDecision(
                SafetyDecisionType.SOURCE_SELECTION,
                SafetyAuthorityLevel.COMMUNICATION,
                {"reason": "Communication test 1"},
            ),
            SafetyDecision(
                SafetyDecisionType.COORDINATION_OVERRIDE,
                SafetyAuthorityLevel.FLIGHT_MODE,
                {"reason": "Flight mode test"},
            ),
        ]

        for decision in decisions:
            await authority_manager.validate_safety_decision(decision)

        # Get audit trail
        audit_trail = authority_manager.get_decision_audit_trail()

        assert len(audit_trail) >= 2

        for entry in audit_trail:
            assert "timestamp" in entry
            assert "decision_type" in entry
            assert "requesting_authority" in entry
            assert "approved" in entry
            assert "response_time_ms" in entry
            assert "details" in entry

    def test_authority_hierarchy_ordering(self, authority_manager):
        """Test [9a] - Authority levels are properly ordered by priority"""
        levels = list(authority_manager.authorities.keys())

        # Verify emergency stop has highest priority (lowest number)
        assert SafetyAuthorityLevel.EMERGENCY_STOP == 1
        assert SafetyAuthorityLevel.SIGNAL == 6

        # Verify ordering is maintained
        sorted_levels = sorted(levels)
        assert sorted_levels[0] == SafetyAuthorityLevel.EMERGENCY_STOP
        assert sorted_levels[-1] == SafetyAuthorityLevel.SIGNAL

    @pytest.mark.asyncio
    async def test_response_time_monitoring(self, authority_manager):
        """Test [9a] - Response times are monitored for compliance"""
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
            details={"reason": "Response time test"},
        )

        # Add small delay to test timing
        await asyncio.sleep(0.05)  # 50ms delay to ensure measurable timing

        approved, reason, approving_authority = await authority_manager.validate_safety_decision(
            decision
        )

        assert approved is True
        assert decision.response_time_ms is not None
        assert decision.response_time_ms >= 0  # Should be non-negative
        assert decision.response_time_ms <= 500  # Still within emergency limit

    @pytest.mark.asyncio
    async def test_authority_trigger_tracking(self, authority_manager):
        """Test [9a] - Authority trigger times are tracked"""
        initial_time = authority_manager.authorities[
            SafetyAuthorityLevel.EMERGENCY_STOP
        ].last_trigger
        assert initial_time is None

        # Trigger emergency decision
        decision = SafetyDecision(
            SafetyDecisionType.EMERGENCY_STOP,
            requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
            details={"reason": "Trigger tracking test"},
        )

        await authority_manager.validate_safety_decision(decision)

        # Verify trigger time was updated
        updated_time = authority_manager.authorities[
            SafetyAuthorityLevel.EMERGENCY_STOP
        ].last_trigger
        assert updated_time is not None
        assert updated_time > datetime.now() - timedelta(seconds=1)

    def test_authority_names_and_capabilities(self, authority_manager):
        """Test [9a] - Authority names and capabilities are properly defined"""
        expected_names = {
            SafetyAuthorityLevel.EMERGENCY_STOP: "Operator Emergency Stop",
            SafetyAuthorityLevel.FLIGHT_MODE: "Flight Mode Monitor",
            SafetyAuthorityLevel.GEOFENCE: "Geofence Boundary Enforcement",
            SafetyAuthorityLevel.BATTERY: "Battery Monitor",
            SafetyAuthorityLevel.COMMUNICATION: "Communication Monitor",
            SafetyAuthorityLevel.SIGNAL: "Signal Monitor",
        }

        for level, expected_name in expected_names.items():
            authority = authority_manager.authorities[level]
            assert authority.name == expected_name
            assert authority.override_capability is not None
            assert authority.coordination_integration is not None
