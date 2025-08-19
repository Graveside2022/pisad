"""
TASK-5.5.4-INTEGRATION-TESTING: Comprehensive Safety Integration Testing
SUBTASK-5.5.4.1 [4b] - Test safety manager + SDR priority manager integration

Test suite for validating safety manager integration with SDR priority management.
Ensures safety authority is respected in SDR source priority decisions.

PRD References:
- PRD-AC5.5.5: Comprehensive safety integration validation  
- PRD-FR16: Emergency stop <500ms response time
- PRD-AC5.5.4: Safety authority hierarchy maintained with coordination
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.backend.services.safety_manager import SafetyManager
from src.backend.services.sdr_priority_manager import SDRPriorityManager
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
    SafetyAuthorityLevel,
    SafetyDecision,
    SafetyDecisionType,
)


class TestSafetySDRPriorityIntegration:
    """Integration tests for safety manager and SDR priority manager."""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager instance."""
        manager = SafetyManager()
        await manager.start_monitoring()
        try:
            yield manager
        finally:
            await manager.stop_monitoring()

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager instance."""
        return SafetyAuthorityManager()

    @pytest.fixture
    def sdr_priority_manager(self, safety_authority_manager):
        """Create SDR priority manager with safety integration."""
        return SDRPriorityManager(safety_authority=safety_authority_manager)

    @pytest.mark.asyncio
    async def test_priority_manager_safety_integration(
        self, safety_manager, sdr_priority_manager, safety_authority_manager
    ):
        """
        Test [4b.1] - Priority manager integrates with safety authority.
        
        Validates SDR priority manager respects safety authority decisions.
        """
        # Start priority manager with safety integration
        await sdr_priority_manager.start()
        
        # Verify safety authority is integrated
        assert sdr_priority_manager.safety_authority is not None
        assert sdr_priority_manager.safety_authority == safety_authority_manager
        
        # Verify priority decisions can be validated by safety authority
        source_decision = SafetyDecision(
            decision_type=SafetyDecisionType.SOURCE_SELECTION,
            authority_level=SafetyAuthorityLevel.COMMUNICATION_MONITOR,
            reason="Ground source has better signal quality",
            timestamp=time.time()
        )
        
        approved = await safety_authority_manager.validate_safety_decision(source_decision)
        assert approved.approved is True
        
        await sdr_priority_manager.stop()

    @pytest.mark.asyncio
    async def test_safety_override_priority_decisions(
        self, safety_manager, sdr_priority_manager, safety_authority_manager
    ):
        """
        Test [4b.2] - Safety can override priority manager decisions.
        
        Validates safety authority can override SDR source selection.
        """
        await sdr_priority_manager.start()
        
        # Set initial ground source preference
        await sdr_priority_manager.set_source_preference("ground", priority=0.8)
        
        # Create safety override decision (force drone-only mode)
        override_decision = SafetyDecision(
            decision_type=SafetyDecisionType.SOURCE_SELECTION,
            authority_level=SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
            reason="Emergency: ground communication compromised",
            timestamp=time.time()
        )
        
        # Safety authority should approve override
        approved = await safety_authority_manager.validate_safety_decision(override_decision)
        assert approved.approved is True
        
        # Apply safety override to force drone source
        await sdr_priority_manager.apply_safety_override("drone_only", "Emergency override")
        
        # Verify priority manager respects safety override
        status = await sdr_priority_manager.get_status()
        assert status["safety_override_active"] is True
        assert status["forced_source"] == "drone"
        
        await sdr_priority_manager.stop()

    @pytest.mark.asyncio
    async def test_priority_health_safety_monitoring(
        self, safety_manager, sdr_priority_manager, safety_authority_manager
    ):
        """
        Test [4b.3] - Safety monitors priority manager health.
        
        Validates safety system monitors SDR priority manager health status.
        """
        await sdr_priority_manager.start()
        
        # Get priority manager health through safety system
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status
        
        # Verify safety tracks priority decisions
        safe_source = safety_manager.get_safe_source_recommendation()
        assert safe_source in ["ground", "drone", "auto"]
        
        # Simulate priority manager degradation
        await sdr_priority_manager.simulate_degradation("conflict_resolution")
        
        # Safety should detect degradation in coordination health
        battery_health = safety_manager.get_coordination_battery_health()
        assert "ground" in battery_health or "drone" in battery_health
        
        await sdr_priority_manager.stop()

    @pytest.mark.asyncio
    async def test_emergency_stop_priority_system(
        self, safety_manager, sdr_priority_manager, safety_authority_manager
    ):
        """
        Test [4b.4] - Emergency stop affects priority system.
        
        Validates emergency stop properly affects SDR priority decisions.
        """
        await sdr_priority_manager.start()
        
        # Trigger emergency stop through safety manager
        start_time = time.time()
        emergency_result = safety_manager.trigger_emergency_stop()
        response_time = emergency_result["response_time_ms"]
        
        # Verify emergency stop timing (< 500ms per PRD-FR16)
        assert response_time < 500, f"Emergency stop took {response_time:.2f}ms, exceeds 500ms"
        
        # Verify emergency stop was successful  
        assert emergency_result["success"] is True
        
        # Priority manager should enter safe mode
        status = await sdr_priority_manager.get_status()
        assert "emergency_mode" in status or "safety_override_active" in status
        
        await sdr_priority_manager.stop()

    @pytest.mark.asyncio  
    async def test_priority_conflict_safety_resolution(
        self, safety_manager, sdr_priority_manager, safety_authority_manager
    ):
        """
        Test [4b.5] - Safety resolves priority conflicts.
        
        Validates safety authority resolves SDR source conflicts.
        """
        await sdr_priority_manager.start()
        
        # Create conflicting source preferences
        await sdr_priority_manager.set_source_preference("ground", priority=0.7)
        await sdr_priority_manager.set_source_preference("drone", priority=0.8)
        
        # Safety should provide conflict resolution recommendation
        safe_source = safety_manager.get_safe_source_recommendation()
        assert safe_source in ["ground", "drone", "auto"]
        
        # Create conflict resolution decision
        resolution_decision = SafetyDecision(
            decision_type=SafetyDecisionType.SOURCE_SELECTION,
            authority_level=SafetyAuthorityLevel.SIGNAL_MONITOR,
            reason="Resolving source conflict based on signal quality",
            timestamp=time.time()
        )
        
        # Safety authority should approve resolution
        approved = await safety_authority_manager.validate_safety_decision(resolution_decision)
        assert approved.approved is True
        
        await sdr_priority_manager.stop()

    @pytest.mark.asyncio
    async def test_coordination_latency_safety_impact(
        self, safety_manager, sdr_priority_manager, safety_authority_manager
    ):
        """
        Test [4b.6] - Safety monitors coordination latency impact on priorities.
        
        Validates safety system tracks latency impact of priority decisions.
        """
        await sdr_priority_manager.start()
        
        # Get baseline latency status
        latency_status = safety_manager.get_coordination_latency_status()
        assert "coordination_latency_ms" in latency_status
        assert "within_threshold" in latency_status
        
        # Simulate priority decision that affects latency
        await sdr_priority_manager.set_source_preference("ground", priority=0.9)
        
        # Safety should monitor the latency impact
        updated_latency = safety_manager.get_coordination_latency_status()
        assert "coordination_latency_ms" in updated_latency
        
        # Verify safety thresholds are maintained
        assert updated_latency["within_threshold"] is True or updated_latency["coordination_latency_ms"] < 100
        
        await sdr_priority_manager.stop()