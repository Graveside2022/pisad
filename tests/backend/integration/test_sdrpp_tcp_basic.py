"""
Basic TCP Integration Tests for SDR++ Communication

Simple tests to verify basic functionality without hanging issues.
Following TDD methodology - testing current implementation state.
"""

import asyncio
import socket
from unittest.mock import MagicMock
import pytest
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.services.signal_processor import SignalProcessor


class TestSDRPPBasicTCP:
    """Basic TCP integration tests."""

    @pytest.fixture
    def unused_tcp_port(self):
        """Provide unused TCP port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @pytest.fixture
    async def sdrpp_service(self, unused_tcp_port):
        """Provide basic SDRPPBridgeService instance."""
        service = SDRPPBridgeService()
        service.port = unused_tcp_port
        
        # Mock signal processor
        mock_processor = MagicMock(spec=SignalProcessor)
        mock_processor.get_current_rssi.return_value = -55.2
        service.set_signal_processor(mock_processor)
        
        yield service
        
        if service.running:
            await service.stop()

    @pytest.mark.asyncio
    async def test_server_starts_successfully(self, sdrpp_service):
        """Test that TCP server starts without errors."""
        assert sdrpp_service.running is False
        
        await sdrpp_service.start()
        
        assert sdrpp_service.running is True
        assert sdrpp_service.server is not None

    @pytest.mark.asyncio
    async def test_client_can_connect_and_disconnect(self, sdrpp_service):
        """Test basic client connection and disconnection."""
        await sdrpp_service.start()
        
        # Connect client
        reader, writer = await asyncio.open_connection(
            sdrpp_service.host, sdrpp_service.port
        )
        
        # Verify connection is tracked
        assert len(sdrpp_service.clients) == 1
        
        # Disconnect cleanly
        writer.close()
        await writer.wait_closed()
        
        # Give server time to process disconnection
        await asyncio.sleep(0.2)
        
        # Verify cleanup
        assert len(sdrpp_service.clients) == 0

    @pytest.mark.asyncio
    async def test_message_parsing_method_exists(self, sdrpp_service):
        """Test that message parsing method exists (part of existing implementation)."""
        # This should pass as the method exists
        assert hasattr(sdrpp_service, '_parse_message')
        
        # Test with valid JSON
        valid_json = '{"type": "rssi_update", "timestamp": "2025-08-18T23:00:00Z", "data": {}, "sequence": 1}'
        result = sdrpp_service._parse_message(valid_json)
        
        assert result is not None
        assert result["type"] == "rssi_update"
        
        # Test with invalid JSON
        invalid_json = '{"invalid": json'
        result = sdrpp_service._parse_message(invalid_json)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_rssi_request_handler_exists(self, sdrpp_service):
        """Test that RSSI request handler exists and works."""
        # This should pass as the method exists
        assert hasattr(sdrpp_service, 'handle_rssi_request')
        
        response = await sdrpp_service.handle_rssi_request()
        
        assert "type" in response
        assert response["type"] == "rssi_update"
        assert "data" in response
        assert "rssi" in response["data"]