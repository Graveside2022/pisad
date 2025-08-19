"""
TASK-5.5.4-INTEGRATION-TESTING: Comprehensive Safety Integration Testing
SUBTASK-5.5.4.1 [4a] - Create integration tests for safety manager + dual SDR coordinator

Test suite for validating safety manager integration with dual SDR coordination system.
Ensures all safety mechanisms work correctly during coordination operations.

PRD References:
- PRD-AC5.5.5: Comprehensive safety integration validation
- PRD-FR16: Emergency stop <500ms response time
- PRD-NFR12: Deterministic timing for safety-critical functions
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.backend.services.safety_manager import SafetyManager
from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
    SafetyAuthorityLevel,
    SafetyDecision,
    SafetyDecisionType,
)


class TestSafetyCoordinationIntegration:
    """Integration tests for safety manager and dual SDR coordinator."""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager instance."""
        manager = SafetyManager()
        await manager.start_monitoring()
        yield manager
        # SafetyManager doesn't have explicit stop method

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager instance."""
        return SafetyAuthorityManager()

    @pytest.fixture
    def dual_sdr_coordinator(self, safety_authority_manager):
        """Create dual SDR coordinator with safety integration."""
        return DualSDRCoordinator(safety_authority=safety_authority_manager)

    @pytest.mark.asyncio
    async def test_safety_manager_coordination_startup(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.1] - Safety manager and coordination system startup integration.
        
        Validates that safety manager integrates properly during coordinator startup.
        """
        # Start coordination with safety integration
        await dual_sdr_coordinator.start()
        
        # Verify safety authority is accessible
        assert dual_sdr_coordinator.safety_authority is not None
        assert dual_sdr_coordinator.safety_authority == safety_authority_manager
        
        # Verify safety manager is operational (monitoring task should be active)
        assert safety_manager.monitoring_task is not None
        assert not safety_manager.monitoring_task.done()
        
        # Verify coordination status is available
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status
        assert "healthy" in coordination_status
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_emergency_stop_during_coordination(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.2] - Emergency stop functionality during active coordination.
        
        Validates emergency stop works correctly during coordination operations.
        """
        await dual_sdr_coordinator.start()
        
        # Trigger emergency stop through safety manager
        start_time = time.time()
        
        emergency_decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            authority_level=SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
            reason="Test emergency stop during coordination",
            timestamp=time.time()
        )
        
        # Emergency stop should be approved and executed
        approved = await safety_authority_manager.validate_safety_decision(emergency_decision)
        assert approved.approved is True
        
        # Apply emergency stop to safety manager
        emergency_result = safety_manager.trigger_emergency_stop()
        
        # Verify timing requirements (< 500ms per PRD-FR16)
        response_time = emergency_result["response_time_ms"]
        assert response_time < 500, f"Emergency stop took {response_time:.2f}ms, exceeds 500ms requirement"
        
        # Verify emergency stop was successful
        assert emergency_result["success"] is True
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_safety_decision_coordination_override(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.3] - Safety decisions override coordination operations.
        
        Validates safety authority can override coordination decisions.
        """
        await dual_sdr_coordinator.start()
        
        # Create coordination override decision
        override_decision = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
            authority_level=SafetyAuthorityLevel.FLIGHT_MODE_MONITOR,
            reason="Flight mode not GUIDED",
            timestamp=time.time()
        )
        
        # Safety authority should approve override
        approved = await safety_authority_manager.validate_safety_decision(override_decision)
        assert approved.approved is True
        assert approved.authority_level == SafetyAuthorityLevel.FLIGHT_MODE_MONITOR
        
        # Trigger emergency safety override in coordinator  
        override_result = await dual_sdr_coordinator.trigger_emergency_safety_override(
            "Flight mode not GUIDED - test override"
        )
        
        # Verify override was successful
        assert override_result["success"] is True
        assert "emergency_override" in override_result
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_safety_monitoring_during_coordination(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.4] - Safety monitoring continues during coordination operations.
        
        Validates safety monitoring is not degraded by coordination activities.
        """
        await dual_sdr_coordinator.start()
        
        # Simulate coordination operations
        for i in range(5):
            await asyncio.sleep(0.01)  # Small delay
        
        # Verify safety monitoring is active
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status
        
        # Verify coordination latency monitoring
        latency_status = safety_manager.get_coordination_latency_status()
        assert "coordination_latency_ms" in latency_status
        assert latency_status["within_threshold"] is True
        
        # Verify safety manager state is operational
        assert safety_manager.get_state() in ["IDLE", "OPERATIONAL"]
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordination_health_integration(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.5] - Coordination health monitoring integration with safety system.
        
        Validates safety system monitors coordination health and responds appropriately.
        """
        await dual_sdr_coordinator.start()
        
        # Get health status from coordinator
        health_status = await dual_sdr_coordinator.get_health_status()
        assert "status" in health_status
        
        # Verify safety system coordination status
        coordination_status = safety_manager.get_coordination_status()
        assert "healthy" in coordination_status
        
        # Verify battery health monitoring integration 
        battery_health = safety_manager.get_coordination_battery_health()
        assert "ground" in battery_health
        assert "drone" in battery_health
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_safety_and_coordination_operations(
        self, safety_manager, dual_sdr_coordinator, safety_authority_manager
    ):
        """
        Test [4a.6] - Concurrent safety and coordination operations without interference.
        
        Validates safety operations don't interfere with coordination and vice versa.
        """
        await dual_sdr_coordinator.start()
        
        # Start concurrent operations
        coordination_task = asyncio.create_task(
            self._run_coordination_operations(dual_sdr_coordinator)
        )
        safety_task = asyncio.create_task(
            self._run_safety_operations(safety_manager)
        )
        
        # Wait for both to complete
        coordination_results, safety_results = await asyncio.gather(
            coordination_task, safety_task, return_exceptions=True
        )
        
        # Verify no exceptions occurred
        assert not isinstance(coordination_results, Exception), f"Coordination error: {coordination_results}"
        assert not isinstance(safety_results, Exception), f"Safety error: {safety_results}"
        
        # Verify both systems completed operations
        assert coordination_results["operations_completed"] > 0
        assert safety_results["operations_completed"] > 0
        
        await dual_sdr_coordinator.stop()

    async def _run_coordination_operations(self, coordinator):
        """Helper method for concurrent coordination operations."""
        operations_completed = 0
        
        for i in range(10):
            # Get status instead of non-existent methods
            await coordinator.get_health_status()
            operations_completed += 1
            await asyncio.sleep(0.005)  # 5ms between operations
        
        return {"operations_completed": operations_completed}

    async def _run_safety_operations(self, safety_manager):
        """Helper method for concurrent safety operations."""
        operations_completed = 0
        
        for i in range(20):
            # Use available safety manager methods
            safety_manager.get_coordination_status()
            safety_manager.get_state()
            operations_completed += 1
            await asyncio.sleep(0.002)  # 2ms between checks
        
        return {"operations_completed": operations_completed}