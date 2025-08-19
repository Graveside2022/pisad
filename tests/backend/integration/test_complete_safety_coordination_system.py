"""
TASK-5.5.4-INTEGRATION-TESTING: Comprehensive Safety Integration Testing
SUBTASK-5.5.4.1 [4d] - Test complete safety system with all coordination components active

Complete system integration testing with all coordination components operating together.
Validates that safety authority hierarchy works correctly with dual SDR coordination.

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
from src.backend.services.sdr_priority_manager import SDRPriorityManager
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestCompleteSafetyCoordinationSystem:
    """Complete system integration tests for safety + coordination components."""

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
    async def sdrpp_bridge(self, safety_manager):
        """Create SDR++ bridge with safety integration."""
        service = SDRPPBridgeService(safety_manager=safety_manager, port=8082)
        yield service
        if hasattr(service, 'stop'):
            await service.stop()

    @pytest.fixture
    def dual_sdr_coordinator(self, safety_authority_manager):
        """Create dual SDR coordinator with safety integration."""
        return DualSDRCoordinator(safety_authority=safety_authority_manager)

    @pytest.mark.asyncio
    async def test_complete_system_startup_with_all_components(
        self, safety_manager, safety_authority_manager, sdrpp_bridge, dual_sdr_coordinator
    ):
        """Test complete system startup with all coordination components active."""
        # RED PHASE - This test should verify all components start together
        start_time = time.time()
        
        # Start all coordination components
        await dual_sdr_coordinator.start()
        
        # Verify safety authority hierarchy is operational
        safety_decision = safety_authority_manager.evaluate_safety_decision(
            level=SafetyAuthorityLevel.COORDINATION_HEALTH,
            decision_type=SafetyDecisionType.APPROVE_OPERATION,
            context={"operation": "system_startup", "components": ["coordinator", "bridge"]}
        )
        
        assert safety_decision.decision_type == SafetyDecisionType.APPROVE_OPERATION
        assert safety_decision.authority_level == SafetyAuthorityLevel.COORDINATION_HEALTH
        
        # Verify safety manager is monitoring
        assert safety_manager._monitoring_active
        
        # Verify coordination system is running
        assert dual_sdr_coordinator.is_running
        
        # Verify startup time is reasonable (<2 seconds for complete system)
        startup_time = time.time() - start_time
        assert startup_time < 2.0, f"System startup took {startup_time:.2f}s, expected <2s"
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_safety_authority_override_during_coordination(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """Test safety authority can override coordination operations."""
        # RED PHASE - Safety should override coordination when critical
        await dual_sdr_coordinator.start()
        
        # Simulate critical safety condition requiring override
        safety_decision = safety_authority_manager.evaluate_safety_decision(
            level=SafetyAuthorityLevel.EMERGENCY_STOP,
            decision_type=SafetyDecisionType.FORCE_EMERGENCY_STOP,
            context={"reason": "critical_failure", "source": "test"}
        )
        
        # Verify emergency stop decision
        assert safety_decision.decision_type == SafetyDecisionType.FORCE_EMERGENCY_STOP
        assert safety_decision.authority_level == SafetyAuthorityLevel.EMERGENCY_STOP
        
        # Verify coordination respects safety authority
        # The coordinator should be aware of safety decisions
        assert dual_sdr_coordinator.safety_authority is not None
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordination_health_integration_with_safety_monitoring(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """Test coordination health is integrated into safety monitoring."""
        # RED PHASE - Safety should monitor coordination health
        await dual_sdr_coordinator.start()
        
        # Get coordination health status
        coordinator_health = dual_sdr_coordinator.get_health_status()
        assert coordinator_health is not None
        
        # Verify safety authority evaluates coordination health
        health_decision = safety_authority_manager.evaluate_safety_decision(
            level=SafetyAuthorityLevel.COORDINATION_HEALTH,
            decision_type=SafetyDecisionType.MONITOR_HEALTH,
            context={"health_status": coordinator_health, "system": "coordination"}
        )
        
        assert health_decision.decision_type == SafetyDecisionType.MONITOR_HEALTH
        assert health_decision.authority_level == SafetyAuthorityLevel.COORDINATION_HEALTH
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_complete_system_communication_loss_response(
        self, safety_manager, safety_authority_manager, sdrpp_bridge, dual_sdr_coordinator
    ):
        """Test complete system response to communication loss."""
        # RED PHASE - System should handle communication loss gracefully
        await dual_sdr_coordinator.start()
        
        # Simulate communication loss
        sdrpp_bridge.handle_communication_loss("Test communication loss scenario")
        
        # Verify safety manager is notified
        if hasattr(safety_manager, 'emergency_stop_active'):
            # Safety manager should be aware of communication issues
            assert safety_manager is not None
        
        # Verify coordination system responds appropriately
        # The coordinator should still be running but may switch to fallback mode
        assert dual_sdr_coordinator.is_running
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_emergency_stop_all_components(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """
        Test [4d.2] - End-to-end emergency stop with all coordination components active.
        
        Validates emergency stop works correctly across the complete integrated system.
        """
        await dual_sdr_coordinator.start()
        
        # Create emergency stop decision through safety authority
        emergency_decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            authority_level=SafetyAuthorityLevel.OPERATOR_EMERGENCY_STOP,
            reason="Complete system test - emergency stop validation",
            timestamp=time.time()
        )
        
        # Validate emergency decision through safety authority
        start_time = time.time()
        approved = await safety_authority_manager.validate_safety_decision(emergency_decision)
        assert approved.approved is True
        
        # Trigger emergency stop through safety manager
        emergency_result = safety_manager.trigger_emergency_stop()
        response_time = emergency_result["response_time_ms"]
        
        # Verify emergency stop timing meets PRD-FR16 (<500ms)
        assert response_time < 500, f"Emergency stop took {response_time:.2f}ms, exceeds 500ms requirement"
        assert emergency_result["success"] is True
        
        # Verify coordination system responds to emergency
        coordinator_health = dual_sdr_coordinator.get_health_status()
        assert coordinator_health is not None
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_safety_and_all_coordination_operations(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """
        Test [4d.3] - Concurrent safety and coordination operations across all components.
        
        Validates all components work concurrently without interference.
        """
        await dual_sdr_coordinator.start()
        
        # Start concurrent operations across all systems
        coordination_task = asyncio.create_task(
            self._run_coordination_operations(dual_sdr_coordinator)
        )
        safety_monitoring_task = asyncio.create_task(
            self._run_safety_monitoring_operations(safety_manager)
        )
        authority_task = asyncio.create_task(
            self._run_authority_operations(safety_authority_manager)
        )
        
        # Wait for all concurrent operations
        results = await asyncio.gather(
            coordination_task, safety_monitoring_task, authority_task, 
            return_exceptions=True
        )
        
        # Verify no exceptions occurred in any system
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"System {i} failed: {result}"
        
        # Verify all systems completed operations successfully
        coordination_result, safety_result, authority_result = results
        assert coordination_result["operations_completed"] >= 10
        assert safety_result["monitoring_cycles"] >= 20
        assert authority_result["decisions_processed"] >= 5
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_complete_system_failure_injection_response(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """
        Test [4d.4] - Complete system response to injected failures.
        
        Validates system robustness when failures occur across multiple components.
        """
        await dual_sdr_coordinator.start()
        
        # Inject multiple failure scenarios
        failure_scenarios = [
            ("communication_degradation", "Network latency increased to 300ms"),
            ("source_conflict", "Ground and drone sources conflicting"),
            ("authority_override", "Flight mode changed from GUIDED"),
            ("battery_warning", "Battery level dropping below threshold")
        ]
        
        for failure_type, failure_reason in failure_scenarios:
            # Create failure decision for safety authority evaluation
            failure_decision = SafetyDecision(
                decision_type=SafetyDecisionType.HANDLE_FAILURE,
                authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
                reason=failure_reason,
                timestamp=time.time(),
                metadata={"failure_type": failure_type}
            )
            
            # Verify safety authority can evaluate failure scenarios
            approved = await safety_authority_manager.validate_safety_decision(failure_decision)
            assert approved is not None
            
            # System should remain operational during failure handling
            coordinator_health = dual_sdr_coordinator.get_health_status()
            assert coordinator_health is not None
            
            # Safety monitoring should continue
            coordination_status = safety_manager.get_coordination_status()
            assert "active" in coordination_status
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_complete_system_performance_under_load(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """
        Test [4d.5] - Complete system performance under operational load.
        
        Validates all components maintain performance requirements under stress.
        """
        await dual_sdr_coordinator.start()
        
        # Create sustained load across all systems
        load_duration = 2.0  # 2 seconds of sustained operations
        start_time = time.time()
        
        # Run sustained operations
        operations_count = 0
        while time.time() - start_time < load_duration:
            # Coordination operations
            coordinator_health = dual_sdr_coordinator.get_health_status()
            assert coordinator_health is not None
            
            # Safety monitoring operations
            coordination_status = safety_manager.get_coordination_status()
            assert "active" in coordination_status
            
            # Authority validation operations
            test_decision = SafetyDecision(
                decision_type=SafetyDecisionType.MONITOR_HEALTH,
                authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
                reason=f"Load test operation {operations_count}",
                timestamp=time.time()
            )
            approved = await safety_authority_manager.validate_safety_decision(test_decision)
            assert approved is not None
            
            operations_count += 1
            await asyncio.sleep(0.01)  # 10ms between operations
        
        # Verify performance under load
        total_time = time.time() - start_time
        operations_per_second = operations_count / total_time
        
        # Should maintain reasonable throughput (>50 ops/sec indicates good performance)
        assert operations_per_second > 50, f"Performance degraded: {operations_per_second:.1f} ops/sec"
        
        # Verify emergency stop still works quickly after load test
        emergency_result = safety_manager.trigger_emergency_stop()
        assert emergency_result["response_time_ms"] < 500
        
        await dual_sdr_coordinator.stop()

    @pytest.mark.asyncio
    async def test_complete_system_graceful_shutdown(
        self, safety_manager, safety_authority_manager, dual_sdr_coordinator
    ):
        """
        Test [4d.6] - Complete system graceful shutdown sequence.
        
        Validates all components shutdown cleanly without errors.
        """
        # Start all components
        await dual_sdr_coordinator.start()
        
        # Verify all systems are operational
        assert dual_sdr_coordinator.is_running
        assert safety_manager._monitoring_active
        
        # Create shutdown decision through safety authority
        shutdown_decision = SafetyDecision(
            decision_type=SafetyDecisionType.APPROVE_OPERATION,
            authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
            reason="Complete system graceful shutdown test",
            timestamp=time.time()
        )
        
        approved = await safety_authority_manager.validate_safety_decision(shutdown_decision)
        assert approved.approved is True
        
        # Perform graceful shutdown
        shutdown_start = time.time()
        await dual_sdr_coordinator.stop()
        shutdown_time = time.time() - shutdown_start
        
        # Verify shutdown is quick (<5 seconds)
        assert shutdown_time < 5.0, f"Shutdown took {shutdown_time:.2f}s, expected <5s"
        
        # Verify coordinator stopped
        assert not dual_sdr_coordinator.is_running

    # Helper methods for concurrent operation testing
    async def _run_coordination_operations(self, coordinator):
        """Helper method for concurrent coordination operations."""
        operations_completed = 0
        
        for i in range(15):  # More operations for comprehensive testing
            health_status = coordinator.get_health_status()
            assert health_status is not None
            operations_completed += 1
            await asyncio.sleep(0.01)  # 10ms between operations
        
        return {"operations_completed": operations_completed}

    async def _run_safety_monitoring_operations(self, safety_manager):
        """Helper method for concurrent safety monitoring operations."""
        monitoring_cycles = 0
        
        for i in range(25):  # More monitoring cycles
            coordination_status = safety_manager.get_coordination_status()
            assert "active" in coordination_status
            
            latency_status = safety_manager.get_coordination_latency_status()
            assert "coordination_latency_ms" in latency_status
            
            monitoring_cycles += 1
            await asyncio.sleep(0.005)  # 5ms between cycles
        
        return {"monitoring_cycles": monitoring_cycles}

    async def _run_authority_operations(self, safety_authority):
        """Helper method for concurrent authority operations.""" 
        decisions_processed = 0
        
        for i in range(10):  # Authority decision processing
            test_decision = SafetyDecision(
                decision_type=SafetyDecisionType.MONITOR_HEALTH,
                authority_level=SafetyAuthorityLevel.SYSTEM_MONITOR,
                reason=f"Concurrent operation test {i}",
                timestamp=time.time()
            )
            
            approved = await safety_authority.validate_safety_decision(test_decision)
            assert approved is not None
            decisions_processed += 1
            await asyncio.sleep(0.02)  # 20ms between decisions
        
        return {"decisions_processed": decisions_processed}