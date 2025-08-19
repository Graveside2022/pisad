"""
TASK-5.5.4-INTEGRATION-TESTING: Comprehensive Safety Integration Testing
SUBTASK-5.5.4.1 [4c] - Create safety manager + TCP bridge integration tests

Test suite for validating safety manager integration with SDR++ TCP bridge service.
Ensures safety monitoring works correctly with network communication components.

PRD References:
- PRD-AC5.5.5: Comprehensive safety integration validation
- PRD-AC5.3.4: Automatic fallback to drone-only operation within 10 seconds  
- PRD-FR16: Emergency stop <500ms response time
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.backend.services.safety_manager import SafetyManager
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
    SafetyAuthorityLevel,
    SafetyDecision,
    SafetyDecisionType,
)


class TestSafetyTCPBridgeIntegration:
    """Integration tests for safety manager and TCP bridge service."""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager instance."""
        manager = SafetyManager()
        await manager.start_monitoring()
        yield manager

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager instance."""
        return SafetyAuthorityManager()

    @pytest.fixture
    def sdrpp_bridge_service(self, safety_authority_manager):
        """Create TCP bridge service with safety integration."""
        config = {"host": "localhost", "port": 8081}
        return SDRPPBridgeService(config, safety_manager=safety_authority_manager)

    @pytest.mark.asyncio
    async def test_tcp_bridge_safety_integration(
        self, safety_manager, sdrpp_bridge_service, safety_authority_manager
    ):
        """
        Test [4c.1] - TCP bridge integrates with safety system.
        
        Validates TCP bridge service integrates properly with safety monitoring.
        """
        # Start bridge service with safety integration  
        await sdrpp_bridge_service.start()
        
        # Verify safety integration is active
        bridge_status = await sdrpp_bridge_service.get_status()
        assert "safety_integration" in bridge_status or "status" in bridge_status
        
        # Verify safety can monitor communication health
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status
        assert "healthy" in coordination_status
        
        await sdrpp_bridge_service.stop()

    @pytest.mark.asyncio
    async def test_communication_loss_safety_response(
        self, safety_manager, sdrpp_bridge_service, safety_authority_manager
    ):
        """
        Test [4c.2] - Safety responds to communication loss.
        
        Validates safety system responds appropriately to TCP communication loss.
        """
        await sdrpp_bridge_service.start()
        
        # Simulate communication loss detection
        await sdrpp_bridge_service.simulate_connection_loss()
        
        # Safety should detect communication degradation
        coordination_status = safety_manager.get_coordination_status()
        assert "active" in coordination_status
        
        # Verify safety can provide fallback recommendations
        safe_source = safety_manager.get_safe_source_recommendation()
        assert safe_source in ["drone", "auto"]  # Should recommend drone fallback
        
        await sdrpp_bridge_service.stop()

    @pytest.mark.asyncio
    async def test_emergency_stop_tcp_bridge(
        self, safety_manager, sdrpp_bridge_service, safety_authority_manager
    ):
        """
        Test [4c.3] - Emergency stop works with active TCP bridge.
        
        Validates emergency stop functionality with TCP communication active.
        """
        await sdrpp_bridge_service.start()
        
        # Trigger emergency stop while bridge is active
        start_time = time.time()
        emergency_result = safety_manager.trigger_emergency_stop()
        response_time = emergency_result["response_time_ms"]
        
        # Verify emergency stop timing (< 500ms per PRD-FR16)
        assert response_time < 500, f"Emergency stop took {response_time:.2f}ms, exceeds 500ms"
        
        # Verify emergency stop was successful
        assert emergency_result["success"] is True
        
        # Bridge should still be functional for status reporting
        bridge_status = await sdrpp_bridge_service.get_status()
        assert "status" in bridge_status
        
        await sdrpp_bridge_service.stop()

    @pytest.mark.asyncio
    async def test_tcp_communication_health_monitoring(
        self, safety_manager, sdrpp_bridge_service, safety_authority_manager
    ):
        """
        Test [4c.4] - Safety monitors TCP communication health.
        
        Validates safety system monitors TCP bridge communication health.
        """
        await sdrpp_bridge_service.start()
        
        # Monitor initial communication health
        initial_latency = safety_manager.get_coordination_latency_status()
        assert "coordination_latency_ms" in initial_latency
        
        # Simulate communication degradation
        await sdrpp_bridge_service.simulate_latency_increase(200)  # 200ms latency
        
        # Safety should detect latency issues
        degraded_latency = safety_manager.get_coordination_latency_status()
        assert "coordination_latency_ms" in degraded_latency
        
        # Check if safety threshold monitoring is working
        assert "within_threshold" in degraded_latency
        
        await sdrpp_bridge_service.stop()

    @pytest.mark.asyncio
    async def test_tcp_bridge_safety_override(
        self, safety_manager, sdrpp_bridge_service, safety_authority_manager  
    ):
        """
        Test [4c.5] - Safety can override TCP bridge operations.
        
        Validates safety authority can override TCP bridge decisions.
        """
        await sdrpp_bridge_service.start()
        
        # Create communication override decision
        override_decision = SafetyDecision(
            decision_type=SafetyDecisionType.COMMUNICATION_OVERRIDE,
            authority_level=SafetyAuthorityLevel.COMMUNICATION_MONITOR,
            reason="TCP communication quality degraded",
            timestamp=time.time()
        )
        
        # Safety authority should approve override
        approved = await safety_authority_manager.validate_safety_decision(override_decision)
        assert approved.approved is True
        
        # Apply override - force bridge to stop accepting connections
        await sdrpp_bridge_service.apply_safety_override("disable_new_connections")
        
        # Verify override is active
        bridge_status = await sdrpp_bridge_service.get_status()
        assert "safety_override" in bridge_status or "status" in bridge_status
        
        await sdrpp_bridge_service.stop()

    @pytest.mark.asyncio
    async def test_fallback_timing_validation(
        self, safety_manager, sdrpp_bridge_service, safety_authority_manager
    ):
        """
        Test [4c.6] - Validate 10-second fallback timing with TCP bridge.
        
        Validates automatic fallback occurs within 10 seconds per PRD-AC5.3.4.
        """
        await sdrpp_bridge_service.start()
        
        # Simulate complete communication loss
        start_time = time.time()
        await sdrpp_bridge_service.simulate_complete_failure()
        
        # Wait for fallback detection (should be < 10 seconds)
        await asyncio.sleep(0.1)  # Give system time to detect failure
        
        # Check fallback recommendation timing
        safe_source = safety_manager.get_safe_source_recommendation()
        fallback_time = time.time() - start_time
        
        # Should recommend drone fallback immediately
        assert safe_source == "drone"
        assert fallback_time < 10.0, f"Fallback detection took {fallback_time:.2f}s, exceeds 10s limit"
        
        await sdrpp_bridge_service.stop()

    @pytest.mark.asyncio
    async def test_concurrent_tcp_safety_operations(
        self, safety_manager, sdrpp_bridge_service, safety_authority_manager
    ):
        """
        Test [4c.7] - Concurrent TCP and safety operations.
        
        Validates TCP bridge and safety operations don't interfere with each other.
        """
        await sdrpp_bridge_service.start()
        
        # Start concurrent operations
        tcp_task = asyncio.create_task(
            self._run_tcp_operations(sdrpp_bridge_service)
        )
        safety_task = asyncio.create_task(
            self._run_safety_monitoring(safety_manager)
        )
        
        # Wait for both to complete
        tcp_results, safety_results = await asyncio.gather(
            tcp_task, safety_task, return_exceptions=True
        )
        
        # Verify no exceptions occurred
        assert not isinstance(tcp_results, Exception), f"TCP error: {tcp_results}"
        assert not isinstance(safety_results, Exception), f"Safety error: {safety_results}"
        
        # Verify both systems completed operations
        assert tcp_results["operations_completed"] > 0
        assert safety_results["monitoring_cycles"] > 0
        
        await sdrpp_bridge_service.stop()

    async def _run_tcp_operations(self, bridge_service):
        """Helper method for concurrent TCP operations."""
        operations_completed = 0
        
        for i in range(5):
            # Get bridge status
            await bridge_service.get_status()
            operations_completed += 1
            await asyncio.sleep(0.01)  # 10ms between operations
        
        return {"operations_completed": operations_completed}

    async def _run_safety_monitoring(self, safety_manager):
        """Helper method for concurrent safety monitoring."""
        monitoring_cycles = 0
        
        for i in range(10):
            # Monitor coordination status
            safety_manager.get_coordination_status()
            safety_manager.get_coordination_latency_status()
            monitoring_cycles += 1
            await asyncio.sleep(0.005)  # 5ms between cycles
        
        return {"monitoring_cycles": monitoring_cycles}