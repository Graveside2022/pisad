"""
Test SDR++ Bridge Service Core Functionality

Tests verify authentic TCP server implementation with real integration points.
Following TDD methodology with RED-GREEN-REFACTOR cycles.

PRD References:
- NFR1: Communication reliability (<1% packet loss)
- NFR2: Signal processing latency (<100ms)
- FR9: Enhanced telemetry streaming with dual-SDR coordination
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestSDRPPBridgeServiceCore:
    """Test SDR++ bridge service core functionality with real integration."""

    def test_sdrpp_bridge_service_class_exists(self):
        """RED: Test that SDRPPBridgeService class can be instantiated."""
        # This will fail initially - creating the service class
        service = SDRPPBridgeService()
        assert service is not None
        assert hasattr(service, "__init__")

    def test_sdrpp_bridge_service_has_required_attributes(self):
        """RED: Test service has required attributes for TCP server."""
        service = SDRPPBridgeService()

        # Required attributes for TCP server functionality
        assert hasattr(service, "host")
        assert hasattr(service, "port")
        assert hasattr(service, "server")
        assert hasattr(service, "clients")
        assert hasattr(service, "running")

        # Default values should be set
        assert service.host == "0.0.0.0"  # Listen on all interfaces
        assert service.port == 8081  # SDR++ communication port
        assert service.server is None  # Not started yet
        assert service.clients == []  # No clients connected
        assert service.running is False  # Not running initially

    @pytest.mark.asyncio
    async def test_sdrpp_bridge_service_can_start_server(self):
        """RED: Test that service can start TCP server on port 8081."""
        service = SDRPPBridgeService()

        # Service should have start method
        assert hasattr(service, "start")

        # Start the server
        await service.start()

        # Server should be running
        assert service.running is True
        assert service.server is not None

        # Clean up
        await service.stop()

    @pytest.mark.asyncio
    async def test_sdrpp_bridge_service_can_stop_server(self):
        """RED: Test that service can stop TCP server cleanly."""
        service = SDRPPBridgeService()

        # Service should have stop method
        assert hasattr(service, "stop")

        # Start then stop
        await service.start()
        await service.stop()

        # Server should be stopped
        assert service.running is False
        assert service.server is None

    @pytest.mark.asyncio
    async def test_sdrpp_bridge_service_tracks_client_connections(self):
        """RED: Test that service tracks client connections properly."""
        service = SDRPPBridgeService()

        # Start server
        await service.start()

        # Initially no clients
        assert len(service.clients) == 0

        # Simulate client connection by connecting to the server
        reader, writer = await asyncio.open_connection("localhost", 8081)

        # Give the server a moment to register the connection
        await asyncio.sleep(0.1)

        # Should have one client tracked
        assert len(service.clients) == 1

        # Close client connection
        writer.close()
        await writer.wait_closed()

        # Give the server a moment to clean up
        await asyncio.sleep(0.1)

        # Should be back to zero clients
        assert len(service.clients) == 0

        # Stop server
        await service.stop()

    def test_sdrpp_bridge_service_has_json_message_parser(self):
        """RED: Test that service has JSON message parsing capability."""
        service = SDRPPBridgeService()

        # Should have message parsing method
        assert hasattr(service, "_parse_message")

    def test_json_message_parser_handles_valid_rssi_message(self):
        """RED: Test parsing valid RSSI streaming message."""
        service = SDRPPBridgeService()

        # Valid RSSI message format
        valid_message = '{"type": "rssi_update", "timestamp": "2025-08-18T21:15:00Z", "data": {"rssi": -45.5, "frequency": 2437000000}, "sequence": 1}'

        # Should parse successfully
        parsed = service._parse_message(valid_message)

        assert parsed is not None
        assert parsed["type"] == "rssi_update"
        assert parsed["data"]["rssi"] == -45.5
        assert parsed["sequence"] == 1

    def test_json_message_parser_handles_malformed_json(self):
        """RED: Test that malformed JSON is handled gracefully."""
        service = SDRPPBridgeService()

        # Malformed JSON - missing closing brace
        malformed_message = '{"type": "rssi_update", "timestamp": "2025-08-18T21:15:00Z"'

        # Should return None for malformed JSON
        parsed = service._parse_message(malformed_message)
        assert parsed is None

    def test_json_message_parser_handles_invalid_message_type(self):
        """RED: Test that invalid message types are rejected."""
        service = SDRPPBridgeService()

        # Invalid message type
        invalid_message = (
            '{"type": "invalid_type", "timestamp": "2025-08-18T21:15:00Z", "data": {}}'
        )

        # Should return None for invalid message types
        parsed = service._parse_message(invalid_message)
        assert parsed is None

    def test_sdrpp_bridge_service_has_rssi_handler(self):
        """RED: Test that service has RSSI streaming handler method."""
        service = SDRPPBridgeService()

        # Should have RSSI request handler
        assert hasattr(service, "handle_rssi_request")

    @pytest.mark.asyncio
    async def test_rssi_handler_returns_current_rssi_data(self):
        """RED: Test RSSI handler returns real-time RSSI data in JSON format."""
        service = SDRPPBridgeService()

        # Mock signal processor with realistic RSSI value
        mock_signal_processor = MagicMock()
        mock_signal_processor.get_current_rssi.return_value = -67.5
        service._signal_processor = mock_signal_processor

        # Call RSSI handler
        response = await service.handle_rssi_request()

        # Should return properly formatted JSON response
        assert response is not None
        assert response["type"] == "rssi_update"
        assert "timestamp" in response
        assert response["data"]["rssi"] == -67.5
        assert "sequence" in response

    @pytest.mark.asyncio
    async def test_rssi_handler_handles_signal_processor_unavailable(self):
        """RED: Test RSSI handler gracefully handles signal processor unavailability."""
        service = SDRPPBridgeService()

        # No signal processor configured
        service._signal_processor = None

        # Should handle gracefully and return error response
        response = await service.handle_rssi_request()

        assert response is not None
        assert response["type"] == "error"
        assert "signal_processor_unavailable" in response["data"]["error"]

    def test_sdrpp_bridge_service_can_set_signal_processor(self):
        """RED: Test that service can accept signal processor dependency injection."""
        service = SDRPPBridgeService()
        mock_signal_processor = MagicMock()

        # Should have method to set signal processor
        assert hasattr(service, "set_signal_processor")

        service.set_signal_processor(mock_signal_processor)
        assert service._signal_processor is mock_signal_processor

    def test_sdrpp_bridge_service_has_frequency_control_handler(self):
        """RED: Test that service has frequency control message handler method."""
        service = SDRPPBridgeService()

        # Should have frequency control handler
        assert hasattr(service, "handle_frequency_control")

    @pytest.mark.asyncio
    async def test_frequency_control_handler_sets_valid_frequency(self):
        """RED: Test frequency control handler with valid frequency in PRD range."""
        service = SDRPPBridgeService()

        # Mock signal processor integration service
        mock_signal_processor = MagicMock()
        mock_signal_processor.set_frequency = MagicMock()
        service.set_signal_processor(mock_signal_processor)

        # Valid frequency control message (2.4 GHz within 850 MHz - 6.5 GHz range)
        freq_message = {
            "type": "freq_control",
            "timestamp": "2025-08-18T21:30:00Z",
            "data": {"frequency": 2400000000},  # 2.4 GHz
            "sequence": 42,
        }

        # Call frequency control handler
        response = await service.handle_frequency_control(freq_message)

        # Should call signal processor set_frequency
        mock_signal_processor.set_frequency.assert_called_once_with(2400000000)

        # Should return success response
        assert response is not None
        assert response["type"] == "freq_control_response"
        assert response["data"]["status"] == "success"
        assert response["data"]["frequency"] == 2400000000
        assert "sequence" in response

    @pytest.mark.asyncio
    async def test_frequency_control_handler_rejects_invalid_low_frequency(self):
        """RED: Test frequency control handler rejects frequencies below PRD minimum."""
        service = SDRPPBridgeService()

        # Mock signal processor
        mock_signal_processor = MagicMock()
        service.set_signal_processor(mock_signal_processor)

        # Invalid frequency below 850 MHz
        freq_message = {
            "type": "freq_control",
            "data": {"frequency": 500000000},  # 500 MHz - below minimum
        }

        # Should reject and return error
        response = await service.handle_frequency_control(freq_message)

        assert response is not None
        assert response["type"] == "error"
        assert "frequency_out_of_range" in response["data"]["error"]

        # Should NOT call signal processor
        mock_signal_processor.set_frequency.assert_not_called()

    @pytest.mark.asyncio
    async def test_frequency_control_handler_rejects_invalid_high_frequency(self):
        """RED: Test frequency control handler rejects frequencies above PRD maximum."""
        service = SDRPPBridgeService()

        # Mock signal processor
        mock_signal_processor = MagicMock()
        service.set_signal_processor(mock_signal_processor)

        # Invalid frequency above 6.5 GHz
        freq_message = {
            "type": "freq_control",
            "data": {"frequency": 7000000000},  # 7 GHz - above maximum
        }

        # Should reject and return error
        response = await service.handle_frequency_control(freq_message)

        assert response is not None
        assert response["type"] == "error"
        assert "frequency_out_of_range" in response["data"]["error"]

        # Should NOT call signal processor
        mock_signal_processor.set_frequency.assert_not_called()

    @pytest.mark.asyncio
    async def test_frequency_control_handler_handles_signal_processor_unavailable(self):
        """RED: Test frequency control handler gracefully handles signal processor unavailability."""
        service = SDRPPBridgeService()

        # No signal processor configured
        service._signal_processor = None

        freq_message = {
            "type": "freq_control",
            "data": {"frequency": 2400000000},
        }

        # Should handle gracefully and return error response
        response = await service.handle_frequency_control(freq_message)

        assert response is not None
        assert response["type"] == "error"
        assert "signal_processor_unavailable" in response["data"]["error"]


class TestSDRPPBridgeServiceHeartbeat:
    """Test heartbeat monitoring and timeout detection with authentic TCP connections."""

    def test_sdrpp_bridge_service_has_heartbeat_attributes(self):
        """RED: Test that service has heartbeat tracking attributes."""
        service = SDRPPBridgeService()

        # Should have heartbeat timeout configuration
        assert hasattr(service, "heartbeat_timeout")
        assert service.heartbeat_timeout == 30.0  # 30 seconds for SDR++ connections

        # Should have client heartbeat tracking dictionary
        assert hasattr(service, "client_heartbeats")
        assert isinstance(service.client_heartbeats, dict)
        assert len(service.client_heartbeats) == 0  # Initially empty

    def test_heartbeat_message_type_in_valid_types(self):
        """RED: Test that 'heartbeat' is included in valid message types."""
        service = SDRPPBridgeService()

        # Heartbeat should be a valid message type for JSON protocol
        assert "heartbeat" in service.valid_message_types

    def test_sdrpp_bridge_service_has_heartbeat_handler(self):
        """RED: Test that service has heartbeat message handler method."""
        service = SDRPPBridgeService()

        # Should have heartbeat handler method
        assert hasattr(service, "handle_heartbeat")

    @pytest.mark.asyncio
    async def test_heartbeat_handler_updates_client_timestamp(self):
        """RED: Test heartbeat handler updates client's last heartbeat timestamp."""
        service = SDRPPBridgeService()

        # Mock client address for tracking
        client_addr = ("127.0.0.1", 12345)

        # Heartbeat message from client
        heartbeat_message = {
            "type": "heartbeat",
            "timestamp": "2025-08-18T22:20:00Z",
            "data": {"status": "alive"},
            "sequence": 100,
        }

        # Handle heartbeat message
        response = await service.handle_heartbeat(heartbeat_message, client_addr)

        # Should update client heartbeat timestamp
        assert client_addr in service.client_heartbeats
        assert service.client_heartbeats[client_addr] > 0.0  # Should have timestamp

        # Should return heartbeat acknowledgment response
        assert response is not None
        assert response["type"] == "heartbeat_ack"
        assert "timestamp" in response
        assert "sequence" in response

    @pytest.mark.asyncio
    async def test_heartbeat_timeout_detection_disconnects_client(self):
        """RED: Test that clients are disconnected after heartbeat timeout."""
        service = SDRPPBridgeService()

        # Use a different port to avoid conflicts
        service.port = 8082

        # Start server
        await service.start()

        # Connect a test client
        reader, writer = await asyncio.open_connection("localhost", 8082)

        # Give server time to register client
        await asyncio.sleep(0.1)
        assert len(service.clients) == 1

        # Simulate heartbeat timeout by setting very old timestamp
        import time

        # Use the actual client address that was stored when the connection was made
        actual_client_addr = next(iter(service.client_heartbeats.keys()))
        service.client_heartbeats[actual_client_addr] = time.time() - 35.0  # 35 seconds ago

        # Trigger timeout check (this method doesn't exist yet - will fail)
        await service._check_heartbeat_timeouts()

        # Client should be disconnected after timeout
        await asyncio.sleep(0.1)
        assert len(service.clients) == 0
        assert actual_client_addr not in service.client_heartbeats

        # Clean up
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass  # Ignore cleanup errors
        await service.stop()

    @pytest.mark.asyncio
    async def test_client_tracking_includes_heartbeat_initialization(self):
        """RED: Test that new client connections initialize heartbeat tracking."""
        service = SDRPPBridgeService()

        # Use a different port to avoid conflicts
        service.port = 8083

        # Start server
        await service.start()

        # Connect client
        reader, writer = await asyncio.open_connection("localhost", 8083)

        # Give server time to process connection
        await asyncio.sleep(0.1)

        # Client should be tracked with initial heartbeat timestamp
        assert len(service.clients) == 1
        assert len(service.client_heartbeats) == 1
        # Check that a client heartbeat was recorded (address may differ due to socket details)
        client_heartbeat_addrs = list(service.client_heartbeats.keys())
        assert len(client_heartbeat_addrs) == 1
        assert service.client_heartbeats[client_heartbeat_addrs[0]] > 0.0

        # Clean up
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.1)
        await service.stop()

    def test_sdrpp_bridge_service_has_timeout_check_method(self):
        """RED: Test that service has method to check for heartbeat timeouts."""
        service = SDRPPBridgeService()

        # Should have method to check heartbeat timeouts
        assert hasattr(service, "_check_heartbeat_timeouts")


class TestSDRPPBridgeServiceShutdown:
    """Test graceful shutdown and cleanup procedures with authentic integration testing."""

    def test_sdrpp_bridge_service_has_shutdown_method(self):
        """RED: Test that service has shutdown method for ServiceManager integration."""
        service = SDRPPBridgeService()

        # Should have shutdown method following ServiceManager pattern
        assert hasattr(service, "shutdown")

    @pytest.mark.asyncio
    async def test_shutdown_method_stops_running_server(self):
        """RED: Test shutdown method stops server and sets running flag to False."""
        service = SDRPPBridgeService()

        # Use different port to avoid conflicts
        service.port = 8084

        # Start server first
        await service.start()
        assert service.running is True
        assert service.server is not None

        # Call shutdown method
        await service.shutdown()

        # Server should be stopped
        assert service.running is False
        assert service.server is None

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_all_active_clients(self):
        """RED: Test shutdown gracefully disconnects all connected clients."""
        service = SDRPPBridgeService()

        # Use different port to avoid conflicts
        service.port = 8085

        # Start server
        await service.start()

        # Connect multiple test clients
        client1_reader, client1_writer = await asyncio.open_connection("localhost", 8085)
        client2_reader, client2_writer = await asyncio.open_connection("localhost", 8085)

        # Give server time to register clients
        await asyncio.sleep(0.1)
        assert len(service.clients) == 2
        assert len(service.client_heartbeats) == 2

        # Call shutdown - should disconnect all clients
        await service.shutdown()

        # All clients should be disconnected and tracking cleared
        assert len(service.clients) == 0
        assert len(service.client_heartbeats) == 0

        # Cleanup
        try:
            client1_writer.close()
            client2_writer.close()
            await client1_writer.wait_closed()
            await client2_writer.wait_closed()
        except Exception:
            pass  # Ignore cleanup errors

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors_gracefully(self):
        """RED: Test shutdown handles client disconnection errors gracefully."""
        service = SDRPPBridgeService()

        # Use different port to avoid conflicts
        service.port = 8086

        # Start server
        await service.start()

        # Connect client
        reader, writer = await asyncio.open_connection("localhost", 8086)
        await asyncio.sleep(0.1)

        # Manually close client connection to simulate error condition
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.1)  # Let server detect disconnection

        # Shutdown should complete without raising exceptions
        await service.shutdown()

        # Service should be cleanly stopped
        assert service.running is False
        assert service.server is None
        assert len(service.clients) == 0
        assert len(service.client_heartbeats) == 0

    @pytest.mark.asyncio
    async def test_shutdown_clears_heartbeat_tracking(self):
        """RED: Test shutdown clears all heartbeat tracking data."""
        service = SDRPPBridgeService()

        # Use different port to avoid conflicts
        service.port = 8087

        # Start server and connect client
        await service.start()
        reader, writer = await asyncio.open_connection("localhost", 8087)
        await asyncio.sleep(0.1)

        # Verify client is tracked
        assert len(service.clients) == 1
        assert len(service.client_heartbeats) == 1

        # Shutdown should clear all tracking
        await service.shutdown()

        # All tracking should be cleared
        assert len(service.clients) == 0
        assert len(service.client_heartbeats) == 0
        assert service.running is False

        # Cleanup
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_shutdown_when_not_running_succeeds(self):
        """RED: Test shutdown succeeds even when service not running."""
        service = SDRPPBridgeService()

        # Service not started
        assert service.running is False
        assert service.server is None

        # Shutdown should succeed without errors
        await service.shutdown()

        # State should remain consistent
        assert service.running is False
        assert service.server is None
        assert len(service.clients) == 0
