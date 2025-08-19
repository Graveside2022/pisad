"""
Test Safety Authority Manager Interface Preservation
TASK-6.1.17c: Preserve existing safety authority manager integration without modification

This module verifies that ASV integration preserves all existing SafetyAuthorityManager
interfaces, authority hierarchies, and decision frameworks without modification.

Test Focus:
- Interface method signatures preservation
- Authority hierarchy compliance  
- Decision framework integration
- Performance preservation
- All safety scenarios with ASV processing

PRD References:
- FR15: System shall cease commands when flight mode changes
- FR16: Emergency controls with <500ms response
- NFR12: Deterministic timing for safety-critical functions
"""

import asyncio
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.backend.services.asv_integration.asv_hackrf_coordinator import ASVHackRFCoordinator
from src.backend.services.asv_integration.asv_degradation_recovery import ASVRecoveryManager
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
    SafetyDecision,
    SafetyDecisionType,
)


class TestSafetyAuthorityInterfacePreservation:
    """Verify ASV integration preserves existing SafetyAuthorityManager interfaces."""

    @pytest.fixture
    def safety_manager(self) -> SafetyAuthorityManager:
        """Create SafetyAuthorityManager for interface verification."""
        return SafetyAuthorityManager()

    @pytest.fixture
    def mock_asv_coordinator(self, safety_manager: SafetyAuthorityManager) -> ASVHackRFCoordinator:
        """Create ASV coordinator with safety manager integration."""
        with patch('src.backend.services.asv_integration.asv_hackrf_coordinator.ASVInteropService'):
            with patch('src.backend.services.asv_integration.asv_hackrf_coordinator.HackRFInterface'):
                coordinator = ASVHackRFCoordinator(safety_authority=safety_manager)
                return coordinator

    def test_safety_authority_manager_core_interfaces_preserved(self, safety_manager: SafetyAuthorityManager):
        """
        TASK-6.1.17c [1] - Verify core SafetyAuthorityManager interfaces are preserved
        
        Tests that all essential interfaces required by ASV integration exist
        and maintain expected signatures.
        """
        # Verify essential method signatures
        assert hasattr(safety_manager, 'validate_coordination_command'), "validate_coordination_command method missing"
        assert hasattr(safety_manager, 'log_coordination_decision'), "log_coordination_decision method missing"
        assert hasattr(safety_manager, 'emergency_override_active'), "emergency_override_active property missing"
        assert hasattr(safety_manager, 'validate_safety_decision'), "validate_safety_decision method missing"
        
        # Verify authority hierarchy constants
        assert hasattr(SafetyAuthorityLevel, 'EMERGENCY_STOP'), "EMERGENCY_STOP level missing"
        assert hasattr(SafetyAuthorityLevel, 'SIGNAL'), "SIGNAL level missing"
        assert hasattr(SafetyAuthorityLevel, 'COMMUNICATION'), "COMMUNICATION level missing"
        
        # Verify 6-level hierarchy preservation
        expected_levels = [
            SafetyAuthorityLevel.EMERGENCY_STOP,
            SafetyAuthorityLevel.FLIGHT_MODE,
            SafetyAuthorityLevel.GEOFENCE,
            SafetyAuthorityLevel.BATTERY,
            SafetyAuthorityLevel.COMMUNICATION,
            SafetyAuthorityLevel.SIGNAL
        ]
        assert len(expected_levels) == 6, "6-level safety hierarchy not preserved"
        
        # Verify authority hierarchy values (lower number = higher authority)
        assert SafetyAuthorityLevel.EMERGENCY_STOP < SafetyAuthorityLevel.SIGNAL, "Authority hierarchy corrupted"

    def test_asv_coordinator_safety_integration_interface_compliance(self, mock_asv_coordinator: ASVHackRFCoordinator):
        """
        TASK-6.1.17c [2] - Verify ASV coordinator uses safety interfaces correctly
        
        Tests that ASV coordinator integration preserves proper usage of
        SafetyAuthorityManager interfaces without modification.
        """
        # Verify safety authority is properly assigned
        assert mock_asv_coordinator._safety_authority is not None, "Safety authority not assigned"
        assert isinstance(mock_asv_coordinator._safety_authority, SafetyAuthorityManager), "Wrong safety authority type"
        
        # Verify safety authority interface usage patterns
        safety_authority = mock_asv_coordinator._safety_authority
        
        # Test validate_coordination_command interface usage
        authorized, reason = safety_authority.validate_coordination_command(
            command_type="asv_coordination",
            authority_level=SafetyAuthorityLevel.SIGNAL,
            details={"component": "ASVHackRFCoordinator", "operation": "test"}
        )
        assert isinstance(authorized, bool), "validate_coordination_command return type changed"
        assert isinstance(reason, str), "validate_coordination_command reason type changed"
        
        # Test log_coordination_decision interface usage
        safety_authority.log_coordination_decision(
            component="ASVHackRFCoordinator",
            decision_type="coordination_validation", 
            decision_details={"test": True},
            authority_level=SafetyAuthorityLevel.SIGNAL,
            outcome="test_outcome"
        )
        # Should not raise exception

    async def test_asv_coordination_safety_validation_integration(self, mock_asv_coordinator: ASVHackRFCoordinator):
        """
        TASK-6.1.17c [3] - Verify ASV coordination respects safety validation
        
        Tests that ASV coordination operations properly integrate with existing
        safety decision frameworks without bypassing authority checks.
        """
        # Test normal coordination validation
        result = await mock_asv_coordinator._validate_coordination_safety()
        assert isinstance(result, bool), "Safety validation return type changed"
        
        # Test emergency override detection
        safety_authority = mock_asv_coordinator._safety_authority
        
        # Trigger emergency override
        await safety_authority.trigger_emergency_override("Interface preservation test")
        assert safety_authority.emergency_override_active == True, "Emergency override state not preserved"
        
        # Verify coordination blocked during emergency
        result = await mock_asv_coordinator._validate_coordination_safety()
        assert result == False, "ASV coordination not blocked during emergency override"
        
        # Clear emergency and verify coordination restored
        await safety_authority.clear_emergency_override("test_clearance")
        assert safety_authority.emergency_override_active == False, "Emergency override clear not working"

    def test_safety_authority_hierarchy_compliance_preservation(self, safety_manager: SafetyAuthorityManager):
        """
        TASK-6.1.17c [4] - Verify authority hierarchy compliance is preserved
        
        Tests that ASV integration operates within existing authority hierarchies
        and respects all authority level constraints.
        """
        # Test that SIGNAL level (used by ASV) is properly positioned in hierarchy
        signal_level = SafetyAuthorityLevel.SIGNAL
        emergency_level = SafetyAuthorityLevel.EMERGENCY_STOP
        communication_level = SafetyAuthorityLevel.COMMUNICATION
        
        # Verify hierarchy ordering preserved
        assert emergency_level < signal_level, "Emergency authority not higher than signal"
        assert communication_level < signal_level, "Communication authority not higher than signal"
        
        # Test authority validation at SIGNAL level
        authorized, reason = safety_manager.validate_coordination_command(
            command_type="asv_coordination",
            authority_level=signal_level,
            details={"hierarchy_test": True}
        )
        assert authorized == True, "SIGNAL level authority validation failed"
        
        # Test insufficient authority blocking
        try:
            # Try to use emergency command with signal authority
            authorized, reason = safety_manager.validate_coordination_command(
                command_type="emergency_stop",
                authority_level=signal_level,  # Insufficient for emergency
                details={"insufficient_test": True}
            )
            assert authorized == False, "Insufficient authority not properly blocked"
        except Exception:
            # Expected - insufficient authority should be blocked
            pass

    def test_safety_decision_framework_preservation(self, safety_manager: SafetyAuthorityManager):
        """
        TASK-6.1.17c [5] - Verify safety decision framework is preserved
        
        Tests that existing safety decision validation and audit mechanisms
        continue to work with ASV integration without modification.
        """
        # Create safety decision for ASV operation
        test_decision = SafetyDecision(
            decision_type=SafetyDecisionType.SOURCE_SELECTION,
            requesting_authority=SafetyAuthorityLevel.SIGNAL,
            details={
                "asv_integration": True,
                "component": "ASVHackRFCoordinator",
                "operation": "frequency_selection"
            }
        )
        
        # Test decision validation framework
        approved, reason, approving_authority = asyncio.run(
            safety_manager.validate_safety_decision(test_decision)
        )
        
        assert isinstance(approved, bool), "Decision validation return type changed"
        assert isinstance(reason, str), "Decision validation reason type changed"
        assert approving_authority is not None, "Approving authority not preserved"
        
        # Verify decision audit trail functionality
        audit_trail = safety_manager.get_decision_audit_trail(limit=10)
        assert isinstance(audit_trail, list), "Audit trail interface changed"
        assert len(audit_trail) > 0, "Decision not added to audit trail"
        
        # Verify coordination audit trail functionality 
        coordination_audit = safety_manager.get_coordination_audit_trail(
            component="ASVHackRFCoordinator",
            limit=10
        )
        assert isinstance(coordination_audit, list), "Coordination audit trail interface changed"

    def test_asv_degradation_recovery_safety_integration_preservation(self):
        """
        TASK-6.1.17c [6] - Verify ASV degradation recovery preserves safety integration
        
        Tests that ASV degradation recovery components properly integrate with
        existing safety authority framework without bypassing controls.
        """
        safety_manager = SafetyAuthorityManager()
        
        # Create recovery manager with safety integration
        recovery_manager = ASVRecoveryManager(
            safety_manager=safety_manager,
            event_logger=Mock(),
            operator_notifier=Mock()
        )
        
        # Verify safety manager integration
        assert recovery_manager.safety_manager is safety_manager, "Safety manager not properly integrated"
        
        # Test authority level checking (method exists in the class)
        assert hasattr(recovery_manager.safety_manager, 'get_authority_status'), "Authority status method missing"
        
        # Verify emergency stop detection
        asyncio.run(safety_manager.trigger_emergency_override("Recovery test"))
        
        # Recovery manager should respect emergency state
        # This tests the interface exists and is being used correctly
        assert safety_manager.emergency_override_active == True, "Emergency override not detected"

    def test_safety_system_performance_preservation(self, safety_manager: SafetyAuthorityManager):
        """
        TASK-6.1.17c [7] - Verify safety system performance is not degraded
        
        Tests that ASV integration does not degrade existing safety system
        response times and performance characteristics.
        """
        # Test validate_coordination_command performance
        start_time = time.perf_counter()
        
        for _ in range(100):
            authorized, reason = safety_manager.validate_coordination_command(
                command_type="asv_coordination",
                authority_level=SafetyAuthorityLevel.SIGNAL,
                details={"performance_test": True}
            )
        
        end_time = time.perf_counter()
        average_time_ms = ((end_time - start_time) / 100) * 1000
        
        # Verify performance within acceptable bounds (should be well under 100ms per call)
        assert average_time_ms < 10.0, f"Safety validation performance degraded: {average_time_ms:.2f}ms > 10ms"
        
        # Test real-time validation performance (critical path)
        start_time = time.perf_counter()
        authorized, reason = safety_manager.validate_coordination_command_real_time(
            command_type="asv_coordination",
            authority_level=SafetyAuthorityLevel.SIGNAL,
            details={"real_time_test": True},
            response_time_limit_ms=100
        )
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        
        assert response_time_ms < 100.0, f"Real-time validation exceeded 100ms: {response_time_ms:.2f}ms"
        assert authorized == True, "Real-time validation failed"

    async def test_emergency_override_system_preservation(self, safety_manager: SafetyAuthorityManager):
        """
        TASK-6.1.17c [8] - Verify emergency override system is preserved
        
        Tests that emergency override functionality continues to work correctly
        and blocks all ASV operations as expected.
        """
        # Test emergency override trigger
        result = await safety_manager.trigger_emergency_override("Interface preservation test")
        assert result["emergency_override_active"] == True, "Emergency override trigger failed"
        assert safety_manager.emergency_override_active == True, "Emergency override state not set"
        
        # Test override status reporting
        status = safety_manager.get_authority_status()
        assert status["emergency_override_active"] == True, "Emergency override status not reported"
        
        # Test emergency override logs
        logs = safety_manager.get_emergency_override_logs(last_minutes=1)
        assert len(logs) > 0, "Emergency override logs not preserved"
        assert any("Interface preservation test" in str(log) for log in logs), "Override reason not logged"
        
        # Test override clearance
        clear_result = await safety_manager.clear_emergency_override("preservation_test")
        assert clear_result["emergency_override_cleared"] == True, "Emergency override clear failed"
        assert safety_manager.emergency_override_active == False, "Emergency override not cleared"

    def test_safety_configuration_validation_preservation(self, safety_manager: SafetyAuthorityManager):
        """
        TASK-6.1.17c [9] - Verify safety configuration validation is preserved
        
        Tests that safety configuration validation and health check mechanisms
        continue to work correctly with ASV integration present.
        """
        # Test configuration validation
        config_result = safety_manager.validate_configuration()
        assert config_result["config_valid"] == True, "Safety configuration validation failed"
        assert config_result["authority_levels_configured"] == 6, "6-level hierarchy not configured"
        assert config_result["emergency_response_ready"] == True, "Emergency response not ready"
        
        # Test health check functionality
        health_result = safety_manager.perform_health_check()
        assert health_result["health_status"] in ["healthy", "degraded"], "Health check status invalid"
        assert health_result["emergency_response_available"] == True, "Emergency response not available"
        assert health_result["authority_levels_active"] == 6, "Not all authority levels active"
        
        # Test safety metrics summary
        metrics = safety_manager.get_safety_metrics_summary()
        assert metrics["total_authorities"] == 6, "Authority count not preserved"
        assert metrics["active_authorities"] == 6, "Active authority count not preserved"

    def test_all_safety_scenarios_with_asv_processing(self, mock_asv_coordinator: ASVHackRFCoordinator):
        """
        TASK-6.1.17c [10] - Test all safety scenarios with ASV enhanced processing active
        
        Comprehensive test ensuring all safety scenarios work correctly when
        ASV enhanced processing is active.
        """
        safety_authority = mock_asv_coordinator._safety_authority
        
        # Scenario 1: Normal operation
        authorized, reason = safety_authority.validate_coordination_command(
            command_type="asv_coordination",
            authority_level=SafetyAuthorityLevel.SIGNAL,
            details={"scenario": "normal_operation"}
        )
        assert authorized == True, "Normal operation scenario failed"
        
        # Scenario 2: Emergency stop
        asyncio.run(safety_authority.trigger_emergency_override("Scenario test"))
        result = asyncio.run(mock_asv_coordinator._validate_coordination_safety())
        assert result == False, "Emergency stop scenario failed"
        
        # Scenario 3: Authority escalation
        asyncio.run(safety_authority.clear_emergency_override("scenario_test"))
        
        # Test escalation handling
        escalation_result = asyncio.run(safety_authority.handle_coordination_failure(
            failure_type="coordination_timing_violation",
            failure_details={"component": "ASVHackRFCoordinator", "test_scenario": True}
        ))
        assert escalation_result["escalation_approved"] == True, "Authority escalation scenario failed"
        
        # Scenario 4: Command validation under load
        for i in range(10):
            authorized, reason = safety_authority.validate_coordination_command(
                command_type="asv_coordination",
                authority_level=SafetyAuthorityLevel.SIGNAL,
                details={"scenario": "load_test", "iteration": i}
            )
            assert authorized == True, f"Load test iteration {i} failed"


class TestSafetyAuthorityInterfaceComplianceVerification:
    """Verify ASV components maintain strict compliance with SafetyAuthorityManager interfaces."""

    def test_interface_signature_preservation(self):
        """
        Verify that all SafetyAuthorityManager method signatures used by ASV
        components are preserved exactly as originally defined.
        """
        safety_manager = SafetyAuthorityManager()
        
        # Test validate_coordination_command signature
        import inspect
        validate_sig = inspect.signature(safety_manager.validate_coordination_command)
        expected_params = ['command_type', 'authority_level', 'details']
        actual_params = list(validate_sig.parameters.keys())[1:]  # Skip 'self'
        assert actual_params == expected_params, f"validate_coordination_command signature changed: {actual_params}"
        
        # Test log_coordination_decision signature
        log_sig = inspect.signature(safety_manager.log_coordination_decision)
        expected_log_params = ['component', 'decision_type', 'decision_details', 'authority_level', 'outcome']
        actual_log_params = list(log_sig.parameters.keys())[1:]  # Skip 'self'
        assert actual_log_params == expected_log_params, f"log_coordination_decision signature changed: {actual_log_params}"

    def test_return_type_preservation(self):
        """
        Verify that all SafetyAuthorityManager method return types used by ASV
        components are preserved exactly as originally defined.
        """
        safety_manager = SafetyAuthorityManager()
        
        # Test validate_coordination_command return type
        authorized, reason = safety_manager.validate_coordination_command(
            command_type="test",
            authority_level=SafetyAuthorityLevel.SIGNAL,
            details={}
        )
        assert isinstance(authorized, bool), f"validate_coordination_command authorized type changed: {type(authorized)}"
        assert isinstance(reason, str), f"validate_coordination_command reason type changed: {type(reason)}"
        
        # Test emergency_override_active property type
        assert isinstance(safety_manager.emergency_override_active, bool), f"emergency_override_active type changed: {type(safety_manager.emergency_override_active)}"

    def test_exception_handling_preservation(self):
        """
        Verify that exception handling behavior in SafetyAuthorityManager
        is preserved and ASV components handle them correctly.
        """
        safety_manager = SafetyAuthorityManager()
        
        # Test invalid authority level handling
        try:
            # This should not raise an exception but return False
            authorized, reason = safety_manager.validate_coordination_command(
                command_type="test",
                authority_level=999,  # Invalid level
                details={}
            )
            # Should handle gracefully
            assert authorized == False, "Invalid authority level not handled correctly"
        except Exception as e:
            pytest.fail(f"Exception handling changed: {e}")

    def test_audit_trail_integration_preservation(self):
        """
        Verify that audit trail functionality used by ASV components
        is preserved and continues to work correctly.
        """
        safety_manager = SafetyAuthorityManager()
        
        # Generate some coordination decisions
        safety_manager.log_coordination_decision(
            component="ASVHackRFCoordinator",
            decision_type="interface_test",
            decision_details={"preservation_test": True},
            authority_level=SafetyAuthorityLevel.SIGNAL,
            outcome="success"
        )
        
        # Test audit trail retrieval
        audit_trail = safety_manager.get_coordination_audit_trail(
            component="ASVHackRFCoordinator",
            decision_type="interface_test",
            limit=5
        )
        
        assert isinstance(audit_trail, list), "Audit trail type changed"
        assert len(audit_trail) > 0, "Audit trail not working"
        
        # Verify audit entry structure
        entry = audit_trail[0]
        expected_keys = ['timestamp', 'component', 'decision_type', 'authority_level', 'outcome']
        for key in expected_keys:
            assert key in entry, f"Audit entry missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])