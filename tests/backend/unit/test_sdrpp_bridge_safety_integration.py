"""
Test SDRPPBridgeService safety integration and communication health monitoring.

Tests SUBTASK-5.5.1.3 implementation with steps [1m] through [1r].
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestSDRPPBridgeSafetyIntegration:
    """Test SDRPPBridgeService safety integration features."""

    @pytest.fixture
    def bridge_service(self):
        """Create SDRPPBridgeService instance."""
        return SDRPPBridgeService()

    @pytest.fixture
    def mock_safety_manager(self):
        """Create mock safety manager."""
        safety_manager = AsyncMock()
        safety_manager.handle_communication_loss = AsyncMock()
        safety_manager.handle_communication_restored = AsyncMock()
        return safety_manager

    def test_safety_manager_integration(self, bridge_service, mock_safety_manager):
        """Test [1m] - Safety manager integration for communication health monitoring."""
        # Test setting safety manager
        bridge_service.set_safety_manager(mock_safety_manager)

        assert bridge_service._safety_manager == mock_safety_manager
        assert bridge_service._safety_timeout == 10.0  # <10s timeout requirement
        assert bridge_service._communication_quality_threshold == 0.5

    def test_connection_callbacks_registration(self, bridge_service):
        """Test [1n] - Connection lost/restored callback registration."""
        # Mock callbacks
        lost_callback = MagicMock()
        restored_callback = MagicMock()

        # Register callbacks
        bridge_service.add_connection_lost_callback(lost_callback)
        bridge_service.add_connection_restored_callback(restored_callback)

        assert lost_callback in bridge_service._connection_lost_callbacks
        assert restored_callback in bridge_service._connection_restored_callbacks

    @pytest.mark.asyncio
    async def test_communication_health_status(self, bridge_service):
        """Test [1q] - Communication health status for safety decision matrix."""
        # Initialize metrics
        bridge_service._message_success_count = 80
        bridge_service._message_failure_count = 20
        bridge_service._message_latency_history = [50.0, 75.0, 60.0]  # Average 61.67ms
        bridge_service._connection_start_time = time.time() - 300  # 5 minutes ago
        bridge_service._consecutive_failures = 1

        # Mock active client
        mock_client = MagicMock()
        bridge_service.clients = [mock_client]
        bridge_service.client_heartbeats = {("127.0.0.1", 12345): time.time() - 5}

        health_status = await bridge_service.get_communication_health_status()

        # Verify health status calculation
        assert health_status["connected_clients"] == 1
        assert health_status["message_success_rate"] == 0.8  # 80/100
        assert health_status["average_latency_ms"] == 61.666666666666664
        assert health_status["consecutive_failures"] == 1
        assert health_status["healthy"] is True  # Above threshold
        assert "connection_duration_seconds" in health_status
        assert health_status["last_heartbeat_age"] == pytest.approx(5.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_communication_health_unhealthy(self, bridge_service):
        """Test communication health when conditions are unhealthy."""
        # Set unhealthy conditions
        bridge_service._message_success_count = 30
        bridge_service._message_failure_count = 70  # 30% success rate
        bridge_service._consecutive_failures = 5  # Above failure threshold
        bridge_service.clients = []  # No clients connected

        health_status = await bridge_service.get_communication_health_status()

        assert health_status["healthy"] is False
        assert health_status["connected_clients"] == 0
        assert health_status["message_success_rate"] == 0.3
        assert health_status["consecutive_failures"] == 5

    @pytest.mark.asyncio
    async def test_safety_communication_loss(self, bridge_service, mock_safety_manager):
        """Test [1n] - Communication loss detection with safety event triggers."""
        bridge_service.set_safety_manager(mock_safety_manager)

        # Mock callback functions
        async_callback = AsyncMock()
        sync_callback = MagicMock()
        bridge_service.add_connection_lost_callback(async_callback)
        bridge_service.add_connection_lost_callback(sync_callback)

        # Trigger communication loss
        await bridge_service.safety_communication_loss("Test connection timeout")

        # Verify safety manager was notified
        mock_safety_manager.handle_communication_loss.assert_called_once()
        call_args = mock_safety_manager.handle_communication_loss.call_args[0][0]
        assert call_args["event_type"] == "communication_loss"
        assert call_args["reason"] == "Test connection timeout"
        assert call_args["source"] == "sdrpp_bridge"

        # Verify callbacks were executed
        async_callback.assert_called_once_with("Test connection timeout")
        sync_callback.assert_called_once_with("Test connection timeout")

    @pytest.mark.asyncio
    async def test_safety_communication_restored(self, bridge_service, mock_safety_manager):
        """Test [1n] - Communication restoration with safety notifications."""
        bridge_service.set_safety_manager(mock_safety_manager)
        bridge_service._consecutive_failures = 5

        # Mock callback functions
        async_callback = AsyncMock()
        sync_callback = MagicMock()
        bridge_service.add_connection_restored_callback(async_callback)
        bridge_service.add_connection_restored_callback(sync_callback)

        # Trigger communication restoration
        await bridge_service.safety_communication_restored()

        # Verify failure counter was reset
        assert bridge_service._consecutive_failures == 0
        assert bridge_service._connection_start_time > 0

        # Verify safety manager was notified
        mock_safety_manager.handle_communication_restored.assert_called_once()
        call_args = mock_safety_manager.handle_communication_restored.call_args[0][0]
        assert call_args["event_type"] == "communication_restored"
        assert call_args["source"] == "sdrpp_bridge"

        # Verify callbacks were executed
        async_callback.assert_called_once()
        sync_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_safety_timeout_check(self, bridge_service):
        """Test [1p] - Safety timeout checking with <10s requirement."""
        # Set last connection check to more than 10 seconds ago
        bridge_service._last_connection_check = time.time() - 15.0
        bridge_service.clients = []  # No active connections

        with patch.object(bridge_service, "safety_communication_loss") as mock_loss:
            timeout_exceeded = await bridge_service.check_safety_timeout()

            assert timeout_exceeded is True
            mock_loss.assert_called_once()
            call_args = mock_loss.call_args[0][0]
            assert "Safety timeout exceeded" in call_args
            assert "15.0s" in call_args

    @pytest.mark.asyncio
    async def test_safety_timeout_check_with_clients(self, bridge_service):
        """Test safety timeout check when clients are connected."""
        # Mock active client
        mock_client = MagicMock()
        bridge_service.clients = [mock_client]

        timeout_exceeded = await bridge_service.check_safety_timeout()

        # Should not timeout when clients are connected
        assert timeout_exceeded is False
        assert bridge_service._last_connection_check > 0

    @pytest.mark.asyncio
    async def test_emergency_disconnect(self, bridge_service):
        """Test [1o] - Emergency disconnect for safety-triggered shutdown."""
        # Create mock writers that behave like StreamWriter
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        # Set up mock methods
        mock_client1.is_closing.return_value = False
        mock_client2.is_closing.return_value = False
        mock_client1.write = MagicMock()  # Mock write method
        mock_client2.write = MagicMock()  # Mock write method
        mock_client1.drain = AsyncMock()  # Mock drain method
        mock_client2.drain = AsyncMock()  # Mock drain method
        mock_client1.close = MagicMock()  # Mock close method
        mock_client2.close = MagicMock()  # Mock close method
        mock_client1.wait_closed = AsyncMock()  # Mock wait_closed method
        mock_client2.wait_closed = AsyncMock()  # Mock wait_closed method

        bridge_service.clients = [mock_client1, mock_client2]
        bridge_service.client_heartbeats = {("127.0.0.1", 12345): time.time()}
        bridge_service.running = True

        with patch.object(bridge_service, "safety_communication_loss") as mock_loss:
            await bridge_service.emergency_disconnect("Test emergency")

            # Verify service was stopped
            assert bridge_service.running is False
            assert len(bridge_service.clients) == 0
            assert len(bridge_service.client_heartbeats) == 0

            # Verify emergency messages were sent to both clients
            assert mock_client1.write.called
            assert mock_client2.write.called

            # Verify clients were closed
            mock_client1.close.assert_called_once()
            mock_client2.close.assert_called_once()
            mock_client1.wait_closed.assert_called_once()
            mock_client2.wait_closed.assert_called_once()

            # Verify safety communication loss was triggered
            mock_loss.assert_called_once_with("Emergency disconnect: Test emergency")

    @pytest.mark.asyncio
    async def test_safety_status_integration(self, bridge_service, mock_safety_manager):
        """Test [1r] - Safety status dashboard integration."""
        bridge_service.set_safety_manager(mock_safety_manager)
        bridge_service.add_connection_lost_callback(MagicMock())
        bridge_service.add_connection_restored_callback(MagicMock())

        # Mock health status
        bridge_service._message_success_count = 90
        bridge_service._message_failure_count = 10
        bridge_service._message_latency_history = [25.0, 30.0, 35.0]
        bridge_service._connection_start_time = time.time() - 600  # 10 minutes ago

        # Mock active client
        mock_client = MagicMock()
        bridge_service.clients = [mock_client]

        safety_status = await bridge_service.get_safety_status_integration()

        # Verify communication bridge status
        comm_bridge = safety_status["communication_bridge"]
        assert comm_bridge["status"] == "healthy"
        assert comm_bridge["connected_clients"] == 1
        assert comm_bridge["quality_percentage"] == 90  # 90% success rate
        assert comm_bridge["latency_ms"] == 30.0  # Average of 25, 30, 35
        assert "connection_duration" in comm_bridge
        assert "last_update" in comm_bridge

        # Verify safety integration status
        safety_integration = safety_status["safety_integration"]
        assert safety_integration["safety_manager_connected"] is True
        assert safety_integration["safety_timeout_threshold"] == 10.0
        assert safety_integration["quality_threshold"] == 0.5
        assert safety_integration["callbacks_registered"] == 2

    @pytest.mark.asyncio
    async def test_communication_loss_without_safety_manager(self, bridge_service):
        """Test communication loss handling when no safety manager is set."""
        # Don't set safety manager
        callback = MagicMock()
        bridge_service.add_connection_lost_callback(callback)

        # Should not raise exception without safety manager
        await bridge_service.safety_communication_loss("No safety manager test")

        # Callback should still be executed
        callback.assert_called_once_with("No safety manager test")

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, bridge_service):
        """Test error handling in callbacks."""
        # Mock failing callbacks
        failing_async_callback = AsyncMock(side_effect=Exception("Async callback failed"))
        failing_sync_callback = MagicMock(side_effect=Exception("Sync callback failed"))

        bridge_service.add_connection_lost_callback(failing_async_callback)
        bridge_service.add_connection_lost_callback(failing_sync_callback)

        # Should not raise exception despite callback failures
        await bridge_service.safety_communication_loss("Callback error test")

        # Verify callbacks were attempted
        failing_async_callback.assert_called_once()
        failing_sync_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_safety_manager_communication_failure(self, bridge_service):
        """Test handling when safety manager communication fails."""
        # Mock safety manager that raises exception
        failing_safety_manager = AsyncMock()
        failing_safety_manager.handle_communication_loss.side_effect = Exception(
            "Safety manager failed"
        )

        bridge_service.set_safety_manager(failing_safety_manager)

        # Should not raise exception despite safety manager failure
        await bridge_service.safety_communication_loss("Safety manager failure test")

        # Verify safety manager was attempted
        failing_safety_manager.handle_communication_loss.assert_called_once()

    def test_safety_integration_initialization(self, bridge_service):
        """Test that safety integration components are properly initialized."""
        # Verify safety integration attributes are initialized
        assert bridge_service._safety_manager is None
        assert bridge_service._safety_timeout == 10.0  # <10s requirement
        assert bridge_service._communication_quality_threshold == 0.5
        assert bridge_service._connection_lost_callbacks == []
        assert bridge_service._connection_restored_callbacks == []
        assert bridge_service._consecutive_failures == 0
        assert bridge_service._failure_threshold == 3
        assert bridge_service._message_latency_history == []
        assert bridge_service._message_success_count == 0
        assert bridge_service._message_failure_count == 0

    @pytest.mark.asyncio
    async def test_health_status_calculation_edge_cases(self, bridge_service):
        """Test health status calculation with edge case values."""
        # Test with zero messages (division by zero case)
        bridge_service._message_success_count = 0
        bridge_service._message_failure_count = 0
        bridge_service._message_latency_history = []
        bridge_service._connection_start_time = 0.0

        health_status = await bridge_service.get_communication_health_status()

        # Should handle edge cases gracefully
        assert health_status["healthy"] is False  # No clients, no messages
        assert health_status["connected_clients"] == 0
        assert health_status["message_success_rate"] == 0.0
        assert health_status["average_latency_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_safety_status_integration_error_handling(self, bridge_service):
        """Test error handling in safety status integration."""
        # Force an error by making get_communication_health_status fail
        with patch.object(
            bridge_service,
            "get_communication_health_status",
            side_effect=Exception("Health status failed"),
        ):
            safety_status = await bridge_service.get_safety_status_integration()

            # Should return error status instead of raising exception
            assert safety_status["communication_bridge"]["status"] == "error"
            assert "error" in safety_status["communication_bridge"]
            assert safety_status["safety_integration"]["safety_manager_connected"] is False
