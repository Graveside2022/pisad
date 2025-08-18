"""
Unit tests for DualSDRCoordinator service.

Tests intelligent coordination between ground SDR++ and drone PISAD
signal processing with automatic fallback and safety preservation.

PRD References:
- FR1: Enhanced SDR interface with dual coordination
- FR6: Enhanced RSSI computation with data fusion
- NFR2: <100ms latency maintained through coordination
- NFR12: Deterministic timing for coordination decisions
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator


class TestDualSDRCoordinator:
    """Test cases for DualSDRCoordinator service."""

    @pytest.fixture
    async def coordinator(self):
        """Create coordinator instance for testing."""
        # This test will fail initially - we haven't created the service yet
        coordinator = DualSDRCoordinator()
        yield coordinator

        # Cleanup
        if coordinator.is_running:
            await coordinator.stop()

    @pytest.fixture
    def mock_signal_processor(self):
        """Mock signal processor for testing."""
        processor = MagicMock()
        processor.get_current_rssi = MagicMock(return_value=-50.0)
        processor.get_current_snr = MagicMock(return_value=15.0)
        processor.add_rssi_callback = MagicMock()
        return processor

    @pytest.fixture
    def mock_tcp_bridge(self):
        """Mock TCP bridge service for testing."""
        bridge = MagicMock()
        bridge.is_running = True
        bridge.get_ground_rssi = MagicMock(return_value=-45.0)
        bridge.send_frequency_control = AsyncMock()
        return bridge

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes with proper configuration."""
        # This test will fail - DualSDRCoordinator doesn't exist yet
        assert coordinator is not None
        assert coordinator.coordination_interval == 0.05  # 50ms for <100ms requirement
        assert coordinator.fallback_timeout == 10.0  # 10 seconds as per PRD
        assert coordinator.is_running is False

    @pytest.mark.asyncio
    async def test_frequency_synchronization(
        self, coordinator, mock_signal_processor, mock_tcp_bridge
    ):
        """Test frequency synchronization between ground and drone SDR."""
        # Setup
        coordinator._signal_processor = mock_signal_processor
        coordinator._tcp_bridge = mock_tcp_bridge

        target_frequency = 2.4e9  # 2.4 GHz

        # Test frequency sync
        await coordinator.synchronize_frequency(target_frequency)

        # Verify both systems received frequency update
        mock_signal_processor.set_frequency.assert_called_with(target_frequency)
        mock_tcp_bridge.send_frequency_control.assert_called_with(target_frequency)

    @pytest.mark.asyncio
    async def test_rssi_data_fusion(self, coordinator, mock_signal_processor, mock_tcp_bridge):
        """Test RSSI data fusion selects best signal source."""
        # Setup
        coordinator._signal_processor = mock_signal_processor
        coordinator._tcp_bridge = mock_tcp_bridge

        # Ground has stronger signal (-45 dBm vs -50 dBm)
        mock_tcp_bridge.get_ground_rssi.return_value = -45.0
        mock_signal_processor.get_current_rssi.return_value = -50.0

        # Test fusion logic
        best_rssi = await coordinator.get_best_rssi()

        # Should select ground SDR++ signal
        assert best_rssi == -45.0
        assert coordinator.active_source == "ground"

    @pytest.mark.asyncio
    async def test_automatic_fallback_on_communication_loss(
        self, coordinator, mock_signal_processor, mock_tcp_bridge
    ):
        """Test automatic fallback to drone-only on ground communication loss."""
        # Setup
        coordinator._signal_processor = mock_signal_processor
        coordinator._tcp_bridge = mock_tcp_bridge

        # Simulate communication loss
        mock_tcp_bridge.is_running = False

        # Start coordination loop
        await coordinator.start()

        # Wait for fallback detection
        await asyncio.sleep(0.1)

        # Should fallback to drone-only operation
        assert coordinator.active_source == "drone"
        assert coordinator.fallback_active is True

    @pytest.mark.asyncio
    async def test_coordination_latency_requirement(
        self, coordinator, mock_signal_processor, mock_tcp_bridge
    ):
        """Test coordination decisions meet <50ms latency requirement."""
        # Setup
        coordinator._signal_processor = mock_signal_processor
        coordinator._tcp_bridge = mock_tcp_bridge

        # Measure coordination decision time
        start_time = time.perf_counter()
        await coordinator.make_coordination_decision()
        end_time = time.perf_counter()

        decision_latency = (end_time - start_time) * 1000  # Convert to milliseconds

        # Must meet <50ms requirement for Epic 5 Story 5.3
        assert (
            decision_latency < 50.0
        ), f"Coordination latency {decision_latency:.2f}ms exceeds 50ms requirement"

    @pytest.mark.asyncio
    async def test_signal_strength_comparison(
        self, coordinator, mock_signal_processor, mock_tcp_bridge
    ):
        """Test signal strength comparison and source switching logic."""
        # Setup
        coordinator._signal_processor = mock_signal_processor
        coordinator._tcp_bridge = mock_tcp_bridge

        # Test cases for different signal strength scenarios
        test_cases = [
            (-40.0, -50.0, "ground"),  # Ground stronger
            (-60.0, -45.0, "drone"),  # Drone stronger
            (-50.0, -50.0, "drone"),  # Equal - prefer drone for safety
        ]

        for ground_rssi, drone_rssi, expected_source in test_cases:
            mock_tcp_bridge.get_ground_rssi.return_value = ground_rssi
            mock_signal_processor.get_current_rssi.return_value = drone_rssi

            source = await coordinator.select_best_source()
            assert source == expected_source, f"Failed for ground={ground_rssi}, drone={drone_rssi}"

    @pytest.mark.asyncio
    async def test_health_monitoring_and_status_reporting(
        self, coordinator, mock_signal_processor, mock_tcp_bridge
    ):
        """Test coordination health monitoring and status reporting."""
        # Setup
        coordinator._signal_processor = mock_signal_processor
        coordinator._tcp_bridge = mock_tcp_bridge

        # Get health status
        health_status = await coordinator.get_health_status()

        # Verify comprehensive health reporting
        expected_fields = [
            "coordination_active",
            "active_source",
            "ground_connection_status",
            "drone_signal_quality",
            "coordination_latency_ms",
            "fallback_active",
            "last_decision_timestamp",
        ]

        for field in expected_fields:
            assert field in health_status, f"Missing health status field: {field}"
