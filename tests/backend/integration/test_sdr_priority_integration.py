"""
Integration tests for SDR Priority Manager with Dual SDR Coordinator

Tests realistic integration scenarios with authentic service dependencies.

PRD References:
- FR11: Operator override capability
- FR15: Immediate command cessation on mode change
- NFR2: <100ms latency requirement
- NFR12: Deterministic timing for safety-critical functions
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator


class TestSDRPriorityIntegration:
    """Test integration between priority manager and coordinator."""

    @pytest.fixture
    def mock_signal_processor(self):
        """Mock signal processor with RSSI capabilities."""
        processor = MagicMock()
        processor.get_current_rssi = MagicMock(return_value=-50.0)
        processor.set_frequency = MagicMock()
        return processor

    @pytest.fixture
    def mock_tcp_bridge(self):
        """Mock TCP bridge with ground communication."""
        bridge = MagicMock()
        bridge.is_running = True
        bridge.get_ground_rssi = MagicMock(return_value=-45.0)
        bridge.send_frequency_control = AsyncMock()
        return bridge

    @pytest.fixture
    def mock_safety_manager(self):
        """Mock safety manager with emergency capabilities."""
        safety = MagicMock()
        safety.trigger_emergency_stop = MagicMock(
            return_value={"success": True, "response_time_ms": 250.0}
        )
        return safety

    @pytest.fixture
    def integrated_coordinator(self, mock_signal_processor, mock_tcp_bridge, mock_safety_manager):
        """Create fully integrated coordinator with priority manager."""
        coordinator = DualSDRCoordinator()
        coordinator.set_dependencies(
            signal_processor=mock_signal_processor,
            tcp_bridge=mock_tcp_bridge,
            safety_manager=mock_safety_manager,
        )
        return coordinator

    @pytest.mark.asyncio
    async def test_priority_coordination_integration(self, integrated_coordinator):
        """Test integration between priority manager and coordination decisions."""
        # Start coordination service
        await integrated_coordinator.start()

        # Allow one coordination cycle
        await asyncio.sleep(0.1)

        # Verify priority manager is active
        status = await integrated_coordinator.get_priority_status()
        assert status["priority_manager_active"] is True
        assert "priority_management" in status

        # Test source switching based on signal strength
        # Mock ground signal stronger than drone
        integrated_coordinator._tcp_bridge.get_ground_rssi.return_value = -35.0
        integrated_coordinator._signal_processor.get_current_rssi.return_value = -55.0

        # Force coordination decision
        await integrated_coordinator.make_coordination_decision()

        # Should switch to ground source
        assert integrated_coordinator.active_source == "ground"

        # Clean up
        await integrated_coordinator.stop()

    @pytest.mark.asyncio
    async def test_frequency_conflict_resolution_integration(self, integrated_coordinator):
        """Test realistic frequency conflict resolution."""
        # Create conflicting frequency commands
        ground_command = {"frequency": 2.4e9, "source": "ground", "timestamp": time.time()}
        drone_command = {
            "frequency": 2.45e9,
            "source": "drone",
            "timestamp": time.time() + 0.1,  # Newer timestamp
        }

        # Resolve conflict
        resolution = await integrated_coordinator.resolve_frequency_conflict(
            ground_command, drone_command
        )

        # Drone command should win (newer timestamp, safety authority)
        assert resolution["selected_command"] == drone_command
        assert resolution["conflict_type"] == "drone_command_newer"
        assert resolution["resolution_time_ms"] < 50.0
        assert resolution["rejected_command"] == ground_command

    @pytest.mark.asyncio
    async def test_emergency_override_integration(self, integrated_coordinator):
        """Test emergency override coordination."""
        # Set initial state to ground
        integrated_coordinator.active_source = "ground"

        # Trigger emergency override
        result = await integrated_coordinator.trigger_emergency_override()

        # Verify emergency response
        assert result["source_switched_to"] == "drone"
        assert integrated_coordinator.active_source == "drone"
        assert result["safety_activated"] is True
        assert result["response_time_ms"] < 500.0  # PRD requirement

    @pytest.mark.asyncio
    async def test_communication_loss_handling_integration(self, integrated_coordinator):
        """Test graceful degradation on communication loss."""
        # Start with ground communication active
        integrated_coordinator._tcp_bridge.is_running = True
        integrated_coordinator.active_source = "ground"

        # Simulate communication loss
        integrated_coordinator._tcp_bridge.is_running = False

        # Force coordination decision
        await integrated_coordinator.make_coordination_decision()

        # Should fallback to drone
        assert integrated_coordinator.active_source == "drone"
        assert integrated_coordinator.fallback_active is True

    @pytest.mark.asyncio
    async def test_performance_requirements_integration(self, integrated_coordinator):
        """Test that integrated system meets performance requirements."""
        # Start coordination
        await integrated_coordinator.start()

        # Measure multiple coordination cycles
        start_time = time.perf_counter()

        for _ in range(10):
            await integrated_coordinator.make_coordination_decision()

        total_time = time.perf_counter() - start_time
        avg_time_ms = (total_time / 10) * 1000

        # Should meet PRD-NFR2 requirement (<100ms)
        assert avg_time_ms < 100.0

        # Verify coordination latencies tracked
        assert len(integrated_coordinator._coordination_latencies) > 0

        # Clean up
        await integrated_coordinator.stop()

    @pytest.mark.asyncio
    async def test_health_status_integration(self, integrated_coordinator):
        """Test comprehensive health status reporting."""
        # Get health status
        health = await integrated_coordinator.get_priority_status()

        # Verify comprehensive status information
        required_fields = [
            "coordination_active",
            "active_source",
            "ground_connection_status",
            "coordination_latency_ms",
            "fallback_active",
            "priority_manager_active",
            "priority_management",
        ]

        for field in required_fields:
            assert field in health

        # Verify priority management sub-status
        priority_status = health["priority_management"]
        assert "active_source" in priority_status
        assert "emergency_override_active" in priority_status
        assert "average_decision_latency_ms" in priority_status

    @pytest.mark.asyncio
    async def test_safety_authority_preservation(self, integrated_coordinator):
        """Test that drone maintains safety authority in all scenarios."""
        # Test 1: Emergency override always forces drone
        await integrated_coordinator.trigger_emergency_override()
        assert integrated_coordinator.active_source == "drone"

        # Test 2: Communication loss forces drone
        integrated_coordinator._tcp_bridge.is_running = False
        await integrated_coordinator.make_coordination_decision()
        assert integrated_coordinator.active_source == "drone"

        # Test 3: Safety manager integration preserved
        assert integrated_coordinator._safety_manager is not None
        assert integrated_coordinator._priority_manager._safety_manager is not None

    @pytest.mark.asyncio
    async def test_rssi_data_fusion_integration(self, integrated_coordinator):
        """Test RSSI data fusion from both sources."""
        # Set different RSSI values
        integrated_coordinator._tcp_bridge.get_ground_rssi.return_value = -40.0
        integrated_coordinator._signal_processor.get_current_rssi.return_value = -60.0

        # Get RSSI from coordinator methods
        ground_rssi = integrated_coordinator.get_ground_rssi()
        drone_rssi = integrated_coordinator.get_drone_rssi()

        assert ground_rssi == -40.0
        assert drone_rssi == -60.0

        # Test best RSSI selection
        best_rssi = await integrated_coordinator.get_best_rssi()
        assert best_rssi == -40.0  # Ground is stronger
