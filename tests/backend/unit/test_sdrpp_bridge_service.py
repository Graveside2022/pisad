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
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestSDRPPBridgeServiceCore:
    """Test SDR++ bridge service core functionality with real integration."""

    def test_sdrpp_bridge_service_class_exists(self):
        """RED: Test that SDRPPBridgeService class can be instantiated."""
        # This will fail initially - creating the service class
        service = SDRPPBridgeService()
        assert service is not None
        assert hasattr(service, '__init__')

    def test_sdrpp_bridge_service_has_required_attributes(self):
        """RED: Test service has required attributes for TCP server."""
        service = SDRPPBridgeService()
        
        # Required attributes for TCP server functionality
        assert hasattr(service, 'host')
        assert hasattr(service, 'port')
        assert hasattr(service, 'server')
        assert hasattr(service, 'clients')
        assert hasattr(service, 'running')
        
        # Default values should be set
        assert service.host == "0.0.0.0"  # Listen on all interfaces
        assert service.port == 8081       # SDR++ communication port
        assert service.server is None     # Not started yet
        assert service.clients == []      # No clients connected
        assert service.running is False   # Not running initially

    @pytest.mark.asyncio
    async def test_sdrpp_bridge_service_can_start_server(self):
        """RED: Test that service can start TCP server on port 8081."""
        service = SDRPPBridgeService()
        
        # Service should have start method
        assert hasattr(service, 'start')
        
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
        assert hasattr(service, 'stop')
        
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
        import asyncio
        reader, writer = await asyncio.open_connection('localhost', 8081)
        
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
        assert hasattr(service, '_parse_message')
        
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
        invalid_message = '{"type": "invalid_type", "timestamp": "2025-08-18T21:15:00Z", "data": {}}'
        
        # Should return None for invalid message types
        parsed = service._parse_message(invalid_message)
        assert parsed is None

    def test_sdrpp_bridge_service_has_rssi_handler(self):
        """RED: Test that service has RSSI streaming handler method."""
        service = SDRPPBridgeService()
        
        # Should have RSSI request handler
        assert hasattr(service, 'handle_rssi_request')
        
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
        assert hasattr(service, 'set_signal_processor')
        
        service.set_signal_processor(mock_signal_processor)
        assert service._signal_processor is mock_signal_processor

    def test_sdrpp_bridge_service_has_frequency_control_handler(self):
        """RED: Test that service has frequency control message handler method."""
        service = SDRPPBridgeService()
        
        # Should have frequency control handler
        assert hasattr(service, 'handle_frequency_control')

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
            "sequence": 42
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