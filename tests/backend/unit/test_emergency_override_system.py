"""
Unit tests for Emergency Override System

Tests SUBTASK-5.5.3.3 [10a-10f] implementation - safety-triggered emergency override
in coordination system with bypass pathways.

TDD Phase: RED - Writing failing tests first for emergency override system
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
)


class TestEmergencyOverridePathways:
    """Test suite for emergency override pathways [10a]"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    @pytest.fixture
    def mock_dual_sdr_coordinator(self):
        """Mock DualSDRCoordinator for testing"""
        coordinator = MagicMock()
        coordinator._safety_authority = None
        coordinator.emergency_shutdown = AsyncMock()
        coordinator.force_drone_only_mode = AsyncMock()
        return coordinator

    @pytest.mark.asyncio
    async def test_emergency_bypass_coordination_system(self, authority_manager):
        """Test emergency override bypasses normal coordination pathways"""
        # TDD RED: This test should FAIL initially
        result = await authority_manager.trigger_emergency_coordination_bypass(
            trigger_reason="critical_safety_violation",
            bypass_components=["DualSDRCoordinator", "SDRPriorityManager"],
            fallback_mode="drone_only",
        )

        assert result["bypass_active"] is True
        assert result["bypassed_components"] == ["DualSDRCoordinator", "SDRPriorityManager"]
        assert result["fallback_mode"] == "drone_only"
        assert result["response_time_ms"] < 500  # <500ms requirement

    @pytest.mark.asyncio
    async def test_emergency_override_direct_command_pathway(self, authority_manager):
        """Test direct command pathway that bypasses coordination"""
        # TDD RED: This test should FAIL initially
        result = await authority_manager.execute_emergency_direct_command(
            command_type="switch_to_drone_only",
            target_components=["signal_processor", "state_machine"],
            emergency_reason="coordination_system_failure",
        )

        assert result["executed"] is True
        assert result["bypass_route"] == "direct_emergency_pathway"
        assert "signal_processor" in result["affected_components"]
        assert "state_machine" in result["affected_components"]
        assert result["response_time_ms"] < 100  # Direct pathway should be faster

    @pytest.mark.asyncio
    async def test_emergency_override_with_coordination_isolation(self, authority_manager):
        """Test emergency override isolates coordination system"""
        # TDD RED: This test should FAIL initially
        # First trigger emergency override
        await authority_manager.trigger_emergency_override("test_isolation")

        # Test isolation of coordination system
        isolation_result = await authority_manager.isolate_coordination_system(
            isolation_level="complete", preserve_components=["mavlink_service", "safety_manager"]
        )

        assert isolation_result["isolation_active"] is True
        assert isolation_result["isolated_components"] is not None
        assert "mavlink_service" in isolation_result["preserved_components"]
        assert "safety_manager" in isolation_result["preserved_components"]


class TestCoordinationShutdown:
    """Test suite for immediate coordination shutdown [10b]"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    @pytest.mark.asyncio
    async def test_immediate_coordination_shutdown_critical_trigger(self, authority_manager):
        """Test immediate shutdown on critical safety triggers"""
        # TDD RED: This test should FAIL initially
        shutdown_result = await authority_manager.trigger_immediate_coordination_shutdown(
            safety_trigger="hardware_failure_critical",
            shutdown_level="emergency",
            preserve_emergency_functions=True,
        )

        assert shutdown_result["shutdown_executed"] is True
        assert shutdown_result["shutdown_level"] == "emergency"
        assert shutdown_result["emergency_functions_preserved"] is True
        assert shutdown_result["response_time_ms"] < 200  # Very fast shutdown required

    @pytest.mark.asyncio
    async def test_coordination_shutdown_with_graceful_degradation(self, authority_manager):
        """Test coordination shutdown with graceful degradation"""
        # TDD RED: This test should FAIL initially
        shutdown_result = await authority_manager.trigger_coordinated_shutdown(
            trigger_reason="communication_degradation",
            shutdown_mode="graceful",
            fallback_timeout_ms=1000,
        )

        assert shutdown_result["shutdown_initiated"] is True
        assert shutdown_result["shutdown_mode"] == "graceful"
        assert shutdown_result["fallback_active"] is True
        assert shutdown_result["fallback_timeout_ms"] == 1000


class TestDroneOnlyFallback:
    """Test suite for automatic drone-only mode switching [10c]"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    @pytest.mark.asyncio
    async def test_automatic_drone_only_mode_trigger(self, authority_manager):
        """Test automatic switching to drone-only mode"""
        # TDD RED: This test should FAIL initially
        fallback_result = await authority_manager.trigger_automatic_drone_only_fallback(
            trigger_condition="ground_communication_loss",
            fallback_duration_minutes=10,
            auto_recovery=True,
        )

        assert fallback_result["drone_only_active"] is True
        assert fallback_result["trigger_condition"] == "ground_communication_loss"
        assert fallback_result["fallback_duration_minutes"] == 10
        assert fallback_result["auto_recovery_enabled"] is True
        assert fallback_result["activation_time_ms"] < 300  # Fast activation

    @pytest.mark.asyncio
    async def test_drone_only_mode_with_source_isolation(self, authority_manager):
        """Test drone-only mode isolates ground sources"""
        # TDD RED: This test should FAIL initially
        isolation_result = await authority_manager.isolate_ground_sources_for_drone_only(
            isolation_scope="complete", maintain_emergency_link=True
        )

        assert isolation_result["ground_sources_isolated"] is True
        assert isolation_result["emergency_link_maintained"] is True
        assert isolation_result["isolation_scope"] == "complete"


class TestSafetyOverrideStatusReporting:
    """Test suite for safety override status reporting [10e]"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    def test_emergency_override_status_comprehensive_report(self, authority_manager):
        """Test comprehensive emergency override status reporting"""
        # TDD RED: This test should FAIL initially
        status_report = authority_manager.get_emergency_override_status_report()

        assert "override_active" in status_report
        assert "bypass_pathways" in status_report
        assert "isolated_components" in status_report
        assert "fallback_modes" in status_report
        assert "emergency_functions_status" in status_report
        assert "recovery_readiness" in status_report
        assert "response_time_metrics" in status_report

    def test_emergency_override_logging_infrastructure(self, authority_manager):
        """Test emergency override logging infrastructure"""
        # TDD RED: This test should FAIL initially
        # Trigger an override to generate logs
        asyncio.run(authority_manager.trigger_emergency_override("test_logging"))

        logs = authority_manager.get_emergency_override_logs(last_minutes=5)

        assert len(logs) >= 1
        assert logs[0]["event_type"] == "emergency_override_triggered"
        assert logs[0]["trigger_reason"] == "test_logging"
        assert "timestamp" in logs[0]
        assert "response_time_ms" in logs[0]


class TestSafetyOverrideRecovery:
    """Test suite for safety override recovery procedures [10f]"""

    @pytest.fixture
    def authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    @pytest.mark.asyncio
    async def test_automatic_recovery_when_conditions_safe(self, authority_manager):
        """Test automatic recovery when safety conditions return to normal"""
        # TDD RED: This test should FAIL initially
        # First trigger emergency override
        await authority_manager.trigger_emergency_override("test_recovery")

        # Test automatic recovery
        recovery_result = await authority_manager.attempt_automatic_recovery(
            safety_validation_required=True, recovery_timeout_minutes=5
        )

        assert recovery_result["recovery_attempted"] is True
        assert recovery_result["safety_validation_passed"] is not None
        assert recovery_result["recovery_successful"] is not None

    @pytest.mark.asyncio
    async def test_manual_recovery_authorization(self, authority_manager):
        """Test manual recovery with proper authorization"""
        # TDD RED: This test should FAIL initially
        # Trigger override first
        await authority_manager.trigger_emergency_override("test_manual_recovery")

        # Test manual recovery
        manual_recovery = await authority_manager.authorize_manual_recovery(
            authorized_by="safety_officer",
            authorization_code="EMERGENCY_CLEAR_001",
            force_recovery=False,
        )

        assert manual_recovery["recovery_authorized"] is True
        assert manual_recovery["authorized_by"] == "safety_officer"
        assert manual_recovery["authorization_code"] == "EMERGENCY_CLEAR_001"

    @pytest.mark.asyncio
    async def test_progressive_system_recovery(self, authority_manager):
        """Test progressive recovery of coordination system components"""
        # TDD RED: This test should FAIL initially
        recovery_result = await authority_manager.execute_progressive_recovery(
            recovery_phases=["safety_validation", "component_health_check", "coordination_restore"],
            phase_timeout_minutes=2,
            abort_on_failure=True,
        )

        assert "recovery_phases_completed" in recovery_result
        assert "safety_validation" in recovery_result["recovery_phases_completed"]
        assert recovery_result["progressive_recovery_successful"] is not None
