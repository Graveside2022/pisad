"""
Unit tests for safety-first decision making in coordination

Tests SUBTASK-5.5.3.2 [9c] implementation with comprehensive validation
of safety-first decision making in coordination priority choices.

Validates that all coordination decisions are validated against safety
authority hierarchy with <500ms emergency response preserved.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
    SafetyAuthorityLevel,
    SafetyDecision,
    SafetyDecisionType,
)


class TestSafetyFirstCoordination:
    """Unit tests for safety-first coordination decisions"""

    @pytest.fixture
    def safety_authority(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    @pytest.fixture
    def dual_coordinator(self, safety_authority):
        """Create dual coordinator with safety authority"""
        coordinator = DualSDRCoordinator()
        
        # Mock dependencies
        signal_processor = MagicMock()
        signal_processor.get_current_rssi = MagicMock(return_value=10.0)
        
        tcp_bridge = MagicMock()
        tcp_bridge.is_running = True
        tcp_bridge.get_ground_rssi = MagicMock(return_value=15.0)
        
        safety_manager = MagicMock()
        
        coordinator.set_dependencies(
            signal_processor=signal_processor,
            tcp_bridge=tcp_bridge,
            safety_manager=safety_manager,
            safety_authority=safety_authority
        )
        
        return coordinator

    @pytest.mark.asyncio
    async def test_basic_source_selection_safety_validation(self, dual_coordinator):
        """Test [9c] - Basic source selection with safety validation"""
        # Test safety-validated source selection
        result = await dual_coordinator.select_best_source_with_safety_validation()
        
        # Verify safety validation occurred
        assert "selected_source" in result
        assert "safety_validated" in result
        assert "safety_reason" in result
        assert "response_time_ms" in result
        
        # Should be validated since communication level has authority for source selection
        assert result["safety_validated"] is True
        assert result["response_time_ms"] <= 10000  # <10s communication authority requirement
        assert result["safety_compliant"] is True

    @pytest.mark.asyncio
    async def test_source_selection_ground_preferred_with_safety(self, dual_coordinator):
        """Test [9c] - Ground source preferred when better signal with safety validation"""
        # Ensure ground has better signal
        dual_coordinator._tcp_bridge.get_ground_rssi.return_value = 20.0  # Better than drone
        dual_coordinator._signal_processor.get_current_rssi.return_value = 10.0
        
        result = await dual_coordinator.select_best_source_with_safety_validation()
        
        # Should select ground but with safety validation
        assert result["selected_source"] == "ground"
        assert result["safety_validated"] is True
        assert result["response_time_ms"] >= 0  # Should have non-negative response time

    @pytest.mark.asyncio
    async def test_source_selection_drone_fallback_on_safety_rejection(self, dual_coordinator):
        """Test [9c] - Drone fallback when safety authority rejects source selection"""
        # Create a mock SafetyDecision for the rejection
        mock_decision = MagicMock()
        mock_decision.response_time_ms = 50
        mock_decision.approved = False
        
        # Mock safety authority to reject source selection
        async def mock_validate(decision):
            decision.response_time_ms = 50
            decision.approved = False
            return False, "Test safety rejection", None
            
        dual_coordinator._safety_authority.validate_safety_decision = mock_validate
        
        result = await dual_coordinator.select_best_source_with_safety_validation()
        
        # Should fallback to drone despite potential better ground signal
        assert result["selected_source"] == "drone"
        assert result["safety_validated"] is False
        assert "rejected by safety authority" in str(result["safety_reason"])

    @pytest.mark.asyncio
    async def test_emergency_safety_override_response_time(self, dual_coordinator):
        """Test [9c] - Emergency safety override meets <500ms requirement"""
        # Test emergency override timing
        start_time = asyncio.get_running_loop().time()
        
        result = await dual_coordinator.trigger_emergency_safety_override("Unit test emergency")
        
        end_time = asyncio.get_running_loop().time()
        total_time_ms = int((end_time - start_time) * 1000)
        
        # Verify emergency override results
        assert result["emergency_override_active"] is True
        assert result["source_switched_to"] == "drone"
        assert result["fallback_active"] is True
        assert result["coordination_stopped"] is True
        
        # Verify <500ms timing requirement
        assert result["response_time_ms"] <= 500
        assert result["safety_requirement_met"] is True
        assert total_time_ms <= 500  # Total operation time

    @pytest.mark.asyncio
    async def test_emergency_override_with_safety_authority_integration(self, dual_coordinator):
        """Test [9c] - Emergency override integrates with safety authority"""
        # Mock safety authority emergency override
        expected_safety_result = {
            "emergency_override_active": True,
            "response_time_ms": 100,
            "trigger_reason": "Integration test",
        }
        
        dual_coordinator._safety_authority.trigger_emergency_override = AsyncMock(
            return_value=expected_safety_result
        )
        
        result = await dual_coordinator.trigger_emergency_safety_override("Integration test")
        
        # Verify safety authority was called
        dual_coordinator._safety_authority.trigger_emergency_override.assert_called_once_with("Integration test")
        
        # Verify safety authority result is included
        assert result["safety_authority_override"] == expected_safety_result

    @pytest.mark.asyncio
    async def test_source_selection_without_safety_authority(self, dual_coordinator):
        """Test [9c] - Source selection gracefully handles missing safety authority"""
        # Remove safety authority
        dual_coordinator._safety_authority = None
        
        result = await dual_coordinator.select_best_source_with_safety_validation()
        
        # Should still work but without safety validation
        assert result["safety_validated"] is False
        assert "No safety authority manager" in result["safety_reason"]
        assert result["selected_source"] in ["ground", "drone"]  # Should still select a source

    @pytest.mark.asyncio
    async def test_safety_decision_details_include_coordination_context(self, dual_coordinator):
        """Test [9c] - Safety decisions include coordination context for audit"""
        # Capture the safety decision made
        captured_decision = None
        
        async def capture_decision(decision):
            nonlocal captured_decision
            captured_decision = decision
            return True, "Test approval", SafetyAuthorityLevel.COMMUNICATION
        
        dual_coordinator._safety_authority.validate_safety_decision = capture_decision
        
        await dual_coordinator.select_best_source_with_safety_validation()
        
        # Verify decision contains coordination context
        assert captured_decision is not None
        assert captured_decision.decision_type == SafetyDecisionType.SOURCE_SELECTION
        assert captured_decision.requesting_authority == SafetyAuthorityLevel.COMMUNICATION
        
        details = captured_decision.details
        assert "coordination_component" in details
        assert details["coordination_component"] == "DualSDRCoordinator"
        assert "proposed_source" in details
        assert "current_source" in details
        assert "ground_available" in details
        assert "drone_available" in details

    @pytest.mark.asyncio
    async def test_safety_validation_timing_requirements(self, dual_coordinator):
        """Test [9c] - Safety validation meets timing requirements"""
        # Test multiple source selections to verify consistent timing
        results = []
        for i in range(5):
            result = await dual_coordinator.select_best_source_with_safety_validation()
            results.append(result["response_time_ms"])
        
        # All should meet <10s communication authority timing requirement
        for response_time in results:
            assert response_time <= 10000, f"Response time {response_time}ms exceeds 10s limit"
        
        # Average should be much faster for normal operations
        avg_response_time = sum(results) / len(results)
        assert avg_response_time <= 1000, f"Average response time {avg_response_time}ms too slow"

    @pytest.mark.asyncio
    async def test_coordination_loop_integration_with_safety(self, dual_coordinator):
        """Test [9c] - Coordination loop uses safety-validated decisions"""
        # This tests the integration point where the coordination loop
        # would use the safety-validated source selection
        
        # Start with ground source
        dual_coordinator.active_source = "ground"
        
        # Get safety-validated selection
        result = await dual_coordinator.select_best_source_with_safety_validation()
        
        # Verify the result can be used for coordination decisions
        assert result["selected_source"] in ["ground", "drone"]
        assert isinstance(result["safety_validated"], bool)
        assert isinstance(result["response_time_ms"], int)
        
        # If safety validated, the selected source should be usable
        if result["safety_validated"]:
            assert result["safety_compliant"] is True
            assert result["approving_authority"] in [
                SafetyAuthorityLevel.EMERGENCY_STOP,
                SafetyAuthorityLevel.FLIGHT_MODE,
                SafetyAuthorityLevel.GEOFENCE,
                SafetyAuthorityLevel.BATTERY,
                SafetyAuthorityLevel.COMMUNICATION,
                SafetyAuthorityLevel.SIGNAL,
            ]

    @pytest.mark.asyncio
    async def test_emergency_override_disables_coordination(self, dual_coordinator):
        """Test [9c] - Emergency override completely disables coordination"""
        # Set up coordination task mock
        coordination_task = MagicMock()
        dual_coordinator._coordination_task = coordination_task
        
        result = await dual_coordinator.trigger_emergency_safety_override("Test disable coordination")
        
        # Verify coordination was disabled
        assert result["coordination_stopped"] is True
        coordination_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_safety_authority_error_handling(self, dual_coordinator):
        """Test [9c] - Graceful error handling when safety authority fails"""
        # Mock safety authority to raise exception
        dual_coordinator._safety_authority.validate_safety_decision = AsyncMock(
            side_effect=Exception("Safety authority failure")
        )
        
        result = await dual_coordinator.select_best_source_with_safety_validation()
        
        # Should fallback to drone safely
        assert result["selected_source"] == "drone"
        assert result["safety_validated"] is False
        assert "Safety validation error" in result["safety_reason"]

    def test_safety_coordination_configuration(self, dual_coordinator):
        """Test [9c] - Safety coordination is properly configured"""
        # Verify safety authority is properly set
        assert dual_coordinator._safety_authority is not None
        assert isinstance(dual_coordinator._safety_authority, SafetyAuthorityManager)
        
        # Verify dependencies are properly configured
        assert dual_coordinator._signal_processor is not None
        assert dual_coordinator._tcp_bridge is not None
        assert dual_coordinator._safety_manager is not None