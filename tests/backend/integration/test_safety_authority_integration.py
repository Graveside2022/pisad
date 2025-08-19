"""
Integration tests for Safety Authority Manager with coordination system

Tests SUBTASK-5.5.3.2 [9b] implementation with comprehensive validation
of safety override mechanisms working through coordination system.

Validates that safety authority hierarchy is enforced throughout
coordination system operations with <500ms emergency response.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.backend.core.dependencies import ServiceManager, get_safety_authority_manager
from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
    SafetyDecision,
    SafetyDecisionType,
)
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestSafetyAuthorityIntegration:
    """Integration tests for safety authority manager with coordination system"""

    @pytest.fixture
    async def service_manager(self):
        """Create service manager for testing"""
        return ServiceManager()

    @pytest.fixture
    def dual_coordinator(self):
        """Create dual SDR coordinator for testing"""
        coordinator = DualSDRCoordinator()

        # Mock methods for testing
        coordinator.get_health_status = AsyncMock(
            return_value={
                "coordination_active": True,
                "latency_ms": 45.0,
                "ground_connection_status": 0.9,
            }
        )
        coordinator.trigger_emergency_override = AsyncMock(
            return_value={
                "emergency_override_active": True,
                "response_time_ms": 150.0,
            }
        )

        return coordinator

    @pytest.fixture
    def sdrpp_bridge(self):
        """Create SDR++ bridge for testing"""
        bridge = SDRPPBridgeService()
        bridge.is_running = True
        bridge.get_communication_health = AsyncMock(
            return_value={
                "connected": True,
                "latency_ms": 30.0,
                "quality": 0.95,
            }
        )
        return bridge

    def test_service_manager_safety_authority_initialization(self, service_manager):
        """Test [9b] - Safety authority is properly integrated into service manager"""
        # Verify safety authority is first in startup order
        assert service_manager.startup_order[0] == "safety_authority"

        # Verify safety authority is included in startup order
        assert "safety_authority" in service_manager.startup_order

    @pytest.mark.asyncio
    async def test_safety_authority_service_lifecycle(self, service_manager):
        """Test [9b] - Safety authority manager integrates with service lifecycle"""
        # Initialize services
        await service_manager.initialize_services()

        # Verify safety authority service is initialized
        assert service_manager.initialized is True
        assert "safety_authority" in service_manager.services

        safety_authority = service_manager.get_service("safety_authority")
        assert isinstance(safety_authority, SafetyAuthorityManager)

        # Verify safety authority is ready for emergency response
        status = safety_authority.get_authority_status()
        assert status["emergency_override_active"] is False
        assert len(status["authorities"]) == 6

        # Clean up
        await service_manager.shutdown_services()

    @pytest.mark.asyncio
    async def test_safety_authority_health_monitoring(self, service_manager):
        """Test [9b] - Safety authority status is monitored in service health"""
        # Initialize services
        await service_manager.initialize_services()

        # Get service health
        health = await service_manager.get_service_health()

        # Verify safety authority is in health status
        assert "safety_authority" in health["services"]

        safety_health = health["services"]["safety_authority"]
        assert safety_health["status"] == "healthy"
        assert safety_health["emergency_override"] is False
        assert "active_authorities" in safety_health
        assert safety_health["active_authorities"] == 6  # All 6 authorities active

        # Clean up
        await service_manager.shutdown_services()

    @pytest.mark.asyncio
    async def test_emergency_override_affects_service_health(self, service_manager):
        """Test [9b] - Emergency override changes service health status"""
        # Initialize services
        await service_manager.initialize_services()

        safety_authority = service_manager.get_service("safety_authority")

        # Trigger emergency override
        await safety_authority.trigger_emergency_override("Integration test emergency")

        # Get updated health status
        health = await service_manager.get_service_health()
        safety_health = health["services"]["safety_authority"]

        # Verify emergency override is reflected in health
        assert safety_health["status"] == "emergency"
        assert safety_health["emergency_override"] is True

        # Clean up
        await service_manager.shutdown_services()

    @pytest.mark.asyncio
    async def test_safety_authority_dependency_injection(self):
        """Test [9b] - Safety authority manager can be injected as dependency"""
        # Test dependency injection
        safety_authority = await get_safety_authority_manager()

        assert isinstance(safety_authority, SafetyAuthorityManager)
        assert len(safety_authority.authorities) == 6

        # Verify all authority levels are present
        expected_levels = [
            SafetyAuthorityLevel.EMERGENCY_STOP,
            SafetyAuthorityLevel.FLIGHT_MODE,
            SafetyAuthorityLevel.GEOFENCE,
            SafetyAuthorityLevel.BATTERY,
            SafetyAuthorityLevel.COMMUNICATION,
            SafetyAuthorityLevel.SIGNAL,
        ]

        for level in expected_levels:
            assert level in safety_authority.authorities

    @pytest.mark.asyncio
    async def test_coordination_command_validation_integration(self, dual_coordinator):
        """Test [9b] - Coordination commands are validated through safety authority"""
        safety_authority = SafetyAuthorityManager()

        # Test emergency stop command validation
        valid, reason = safety_authority.validate_coordination_command(
            "emergency_stop", SafetyAuthorityLevel.EMERGENCY_STOP, {"reason": "Test emergency"}
        )
        assert valid is True
        assert "authorized" in reason.lower()

        # Test insufficient authority
        valid, reason = safety_authority.validate_coordination_command(
            "emergency_stop", SafetyAuthorityLevel.SIGNAL, {"reason": "Test emergency"}
        )
        assert valid is False
        assert "insufficient authority" in reason.lower()

    @pytest.mark.asyncio
    async def test_safety_decision_validation_with_coordination(
        self, dual_coordinator, sdrpp_bridge
    ):
        """Test [9b] - Safety decisions are validated for coordination operations"""
        safety_authority = SafetyAuthorityManager()

        # Test source selection decision (communication level authority)
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.SOURCE_SELECTION,
            requesting_authority=SafetyAuthorityLevel.COMMUNICATION,
            details={
                "target_source": "drone_only",
                "reason": "Ground communication lost",
                "coordination_component": "DualSDRCoordinator",
            },
        )

        approved, reason, approving_authority = await safety_authority.validate_safety_decision(
            decision
        )

        assert approved is True
        assert approving_authority == SafetyAuthorityLevel.COMMUNICATION
        assert decision.response_time_ms <= 10000  # <10s requirement

        # Verify decision is logged for audit
        audit_trail = safety_authority.get_decision_audit_trail(limit=1)
        assert len(audit_trail) >= 1
        assert audit_trail[0]["decision_type"] == "source_selection"

    @pytest.mark.asyncio
    async def test_emergency_override_coordination_integration(self, dual_coordinator):
        """Test [9b] - Emergency override works through coordination system"""
        safety_authority = SafetyAuthorityManager()

        # Trigger emergency override
        result = await safety_authority.trigger_emergency_override("Coordination integration test")

        assert result["emergency_override_active"] is True
        assert result["response_time_ms"] <= 500  # <500ms emergency requirement

        # Verify all coordination commands are now blocked
        valid, reason = safety_authority.validate_coordination_command(
            "source_switch", SafetyAuthorityLevel.COMMUNICATION, {}
        )
        assert valid is False
        assert "emergency override active" in reason.lower()

        # Clear emergency override
        clear_result = await safety_authority.clear_emergency_override("Test completion")
        assert clear_result["emergency_override_cleared"] is True

    @pytest.mark.asyncio
    async def test_authority_hierarchy_coordination_integration(self):
        """Test [9b] - Authority hierarchy is enforced in coordination decisions"""
        safety_authority = SafetyAuthorityManager()

        # Test flight mode override (level 2)
        decision = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
            requesting_authority=SafetyAuthorityLevel.FLIGHT_MODE,
            details={
                "mode": "MANUAL",
                "reason": "Pilot override",
                "coordination_action": "Block payload commands",
            },
        )

        approved, reason, approving_authority = await safety_authority.validate_safety_decision(
            decision
        )

        assert approved is True
        assert approving_authority == SafetyAuthorityLevel.FLIGHT_MODE
        assert decision.response_time_ms <= 100  # <100ms requirement

        # Test geofence override (insufficient authority from battery level)
        geofence_decision = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
            requesting_authority=SafetyAuthorityLevel.BATTERY,  # Level 4 - insufficient
            details={
                "boundary": "hard_limit",
                "reason": "Battery triggered geofence",
            },
        )

        approved, reason, approving_authority = await safety_authority.validate_safety_decision(
            geofence_decision
        )

        # Battery level can't override coordination (needs level 2 or higher)
        assert approved is False
        assert "insufficient authority" in reason.lower()

    @pytest.mark.asyncio
    async def test_safety_audit_trail_coordination_context(self):
        """Test [9b] - Safety audit trail includes coordination context"""
        safety_authority = SafetyAuthorityManager()

        # Make several coordination-related decisions
        coordination_decisions = [
            SafetyDecision(
                SafetyDecisionType.SOURCE_SELECTION,
                SafetyAuthorityLevel.COMMUNICATION,
                {
                    "coordination_component": "DualSDRCoordinator",
                    "source": "drone_only",
                    "trigger": "ground_communication_loss",
                },
            ),
            SafetyDecision(
                SafetyDecisionType.COORDINATION_OVERRIDE,
                SafetyAuthorityLevel.FLIGHT_MODE,
                {
                    "coordination_component": "SDRPriorityManager",
                    "override_type": "flight_mode_change",
                    "target_mode": "MANUAL",
                },
            ),
        ]

        for decision in coordination_decisions:
            await safety_authority.validate_safety_decision(decision)

        # Get audit trail
        audit_trail = safety_authority.get_decision_audit_trail()

        # Verify coordination context is preserved
        assert len(audit_trail) >= 2

        for entry in audit_trail:
            assert "details" in entry
            if entry["decision_type"] == "source_selection":
                assert "coordination_component" in entry["details"]
                assert entry["details"]["coordination_component"] == "DualSDRCoordinator"
            elif entry["decision_type"] == "coordination_override":
                assert "coordination_component" in entry["details"]
                assert entry["details"]["coordination_component"] == "SDRPriorityManager"

    @pytest.mark.asyncio
    async def test_response_time_requirements_with_coordination(self):
        """Test [9b] - Response time requirements maintained with coordination overhead"""
        safety_authority = SafetyAuthorityManager()

        # Test emergency stop response time
        start_time = asyncio.get_event_loop().time()

        emergency_decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
            details={
                "reason": "Response time test",
                "coordination_components": [
                    "DualSDRCoordinator",
                    "SDRPriorityManager",
                    "SDRPPBridge",
                ],
            },
        )

        approved, reason, approving_authority = await safety_authority.validate_safety_decision(
            emergency_decision
        )

        end_time = asyncio.get_event_loop().time()
        total_time_ms = (end_time - start_time) * 1000

        assert approved is True
        assert total_time_ms <= 500  # <500ms emergency requirement
        assert emergency_decision.response_time_ms <= 500

        # Test flight mode response time
        flight_mode_decision = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
            requesting_authority=SafetyAuthorityLevel.FLIGHT_MODE,
            details={"mode": "GUIDED", "coordination_validation": True},
        )

        start_time = asyncio.get_event_loop().time()
        approved, reason, approving_authority = await safety_authority.validate_safety_decision(
            flight_mode_decision
        )
        end_time = asyncio.get_event_loop().time()

        flight_time_ms = (end_time - start_time) * 1000

        assert approved is True
        assert flight_time_ms <= 100  # <100ms flight mode requirement
        assert flight_mode_decision.response_time_ms <= 100

    def test_safety_authority_coordination_configuration(self):
        """Test [9b] - Safety authority configuration includes coordination integration"""
        safety_authority = SafetyAuthorityManager()

        # Verify coordination integration points are configured
        expected_integrations = {
            SafetyAuthorityLevel.EMERGENCY_STOP: "All coordination components",
            SafetyAuthorityLevel.FLIGHT_MODE: "DualSDRCoordinator decision making",
            SafetyAuthorityLevel.GEOFENCE: "Priority Manager source selection",
            SafetyAuthorityLevel.COMMUNICATION: "SDRPPBridge health monitoring",
            SafetyAuthorityLevel.SIGNAL: "Dual source signal validation",
        }

        for level, expected_integration in expected_integrations.items():
            authority = safety_authority.authorities[level]
            assert authority.integration_point == expected_integration
            assert authority.coordination_integration is not None
            assert len(authority.coordination_integration) > 0
