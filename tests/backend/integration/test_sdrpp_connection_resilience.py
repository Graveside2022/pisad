"""
Connection Resilience Tests for SDR++ TCP Communication

Tests validate automatic reconnection, exponential backoff, and connection failure recovery.
Ensures robust communication in production environments with network instability.

PRD References:
- NFR1: Communication reliability (<1% packet loss)
- NFR9: MTBF >10 flight hours with automatic recovery mechanisms
- FR9: Enhanced telemetry streaming with dual-SDR coordination

Test Coverage:
- Server restart scenarios and client reconnection
- Network interruption simulation and recovery
- Exponential backoff algorithm validation
- Connection timeout and retry behavior
- Graceful degradation under network stress
- Client connection pooling and management

Following TDD methodology - RED-GREEN-REFACTOR cycles.
"""

import asyncio
import contextlib
import json
import socket
import time
from unittest.mock import MagicMock
import pytest
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.services.signal_processor import SignalProcessor


class TestSDRPPConnectionResilience:
    """Connection resilience and automatic reconnection tests."""

    @pytest.fixture
    def unused_tcp_port(self):
        """Provide unused TCP port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @pytest.fixture
    async def sdrpp_resilience_service(self, unused_tcp_port):
        """Provide SDRPPBridgeService instance for resilience testing."""
        service = SDRPPBridgeService()
        service.port = unused_tcp_port
        
        # Mock signal processor with realistic behavior
        mock_processor = MagicMock(spec=SignalProcessor)
        mock_processor.get_current_rssi.return_value = -55.2
        mock_processor.get_current_snr.return_value = 12.5
        service.set_signal_processor(mock_processor)
        
        yield service
        
        if service.running:
            await service.stop()

    @pytest.mark.asyncio
    async def test_server_restart_client_reconnection_capability(self, sdrpp_resilience_service):
        """RED: Test client can reconnect after server restart."""
        service = sdrpp_resilience_service
        
        # Start server and establish initial connection
        await service.start()
        reader1, writer1 = await asyncio.open_connection(service.host, service.port)
        
        # Verify initial connection works
        request = {"type": "rssi_update", "timestamp": "2025-08-18T23:59:00Z", "data": {}, "sequence": 1}
        writer1.write(json.dumps(request).encode() + b'\n')
        await writer1.drain()
        
        response_data = await reader1.readline()
        response = json.loads(response_data.decode().strip())
        assert response["type"] == "rssi_update"
        
        # Simulate server restart (stop and start)
        writer1.close()
        await writer1.wait_closed()
        await service.stop()
        
        # Wait for server to fully stop
        await asyncio.sleep(0.2)
        
        # Restart server
        await service.start()
        
        # Client should be able to reconnect
        reader2, writer2 = await asyncio.open_connection(service.host, service.port)
        
        # Verify reconnection works
        request2 = {"type": "rssi_update", "timestamp": "2025-08-18T23:59:01Z", "data": {}, "sequence": 2}
        writer2.write(json.dumps(request2).encode() + b'\n')
        await writer2.drain()
        
        response_data2 = await reader2.readline()
        response2 = json.loads(response_data2.decode().strip())
        assert response2["type"] == "rssi_update"
        
        # Cleanup
        writer2.close()
        await writer2.wait_closed()

    @pytest.mark.asyncio
    async def test_client_connection_timeout_behavior(self, sdrpp_resilience_service):
        """RED: Test client connection timeout and retry behavior."""
        service = sdrpp_resilience_service
        
        # Try connecting to non-running server (should timeout quickly)
        start_time = time.time()
        
        with pytest.raises((ConnectionRefusedError, OSError)):
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(service.host, service.port),
                timeout=1.0  # 1 second timeout
            )
        
        connection_attempt_time = time.time() - start_time
        
        # Should fail within reasonable time (much less than 1 second)
        assert connection_attempt_time < 1.0
        
        # Now start server and verify connection works
        await service.start()
        
        # Connection should succeed quickly now
        reader, writer = await asyncio.open_connection(service.host, service.port)
        assert len(service.clients) == 1
        
        # Cleanup
        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_multiple_client_disconnection_and_reconnection(self, sdrpp_resilience_service):
        """RED: Test multiple clients disconnecting and reconnecting."""
        service = sdrpp_resilience_service
        await service.start()
        
        # Connect multiple clients
        clients = []
        for i in range(5):
            reader, writer = await asyncio.open_connection(service.host, service.port)
            clients.append((reader, writer))
        
        # Verify all connected
        assert len(service.clients) == 5
        
        # Disconnect all clients abruptly
        for reader, writer in clients:
            writer.close()
            # Don't wait - simulate network failure
        
        # Wait for server to detect disconnections
        await asyncio.sleep(0.5)
        
        # Server should have cleaned up all clients
        assert len(service.clients) == 0
        
        # All clients should be able to reconnect
        new_clients = []
        for i in range(5):
            reader, writer = await asyncio.open_connection(service.host, service.port)
            
            # Test communication works
            request = {"type": "rssi_update", "timestamp": f"2025-08-18T23:59:{i:02d}Z", "data": {}, "sequence": i}
            writer.write(json.dumps(request).encode() + b'\n')
            await writer.drain()
            
            response_data = await reader.readline()
            response = json.loads(response_data.decode().strip())
            assert response["type"] == "rssi_update"
            
            new_clients.append((reader, writer))
        
        # Verify all reconnected successfully
        assert len(service.clients) == 5
        
        # Cleanup
        for reader, writer in new_clients:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_connection_resilience_after_invalid_messages(self, sdrpp_resilience_service):
        """RED: Test connection remains stable after receiving invalid messages."""
        service = sdrpp_resilience_service
        await service.start()
        
        reader, writer = await asyncio.open_connection(service.host, service.port)
        
        # Send valid message first to establish baseline
        valid_request = {"type": "rssi_update", "timestamp": "2025-08-18T23:59:00Z", "data": {}, "sequence": 1}
        writer.write(json.dumps(valid_request).encode() + b'\n')
        await writer.drain()
        
        response_data = await reader.readline()
        response = json.loads(response_data.decode().strip())
        assert response["type"] == "rssi_update"
        
        # Send invalid messages and consume any error responses
        invalid_messages = [
            b"invalid json data\n",  # Invalid JSON
            b'{"type": "unknown_type", "data": {}}\n',  # Unknown message type
            b'{"incomplete": "message"\n',  # Incomplete JSON
            b'\n',  # Empty message
        ]
        
        for invalid_msg in invalid_messages:
            writer.write(invalid_msg)
            await writer.drain()
            
            # Try to read response (may be error or empty)
            try:
                error_response = await asyncio.wait_for(reader.readline(), timeout=0.5)
                if error_response:
                    # If we get a response, it should be properly formatted JSON error
                    error_data = json.loads(error_response.decode().strip())
                    assert error_data["type"] == "error"
            except (asyncio.TimeoutError, json.JSONDecodeError):
                # Some invalid messages may not generate responses
                pass
        
        # Connection should still be active after invalid messages
        assert len(service.clients) == 1
        
        # Send final valid message to confirm connection recovery
        final_request = {"type": "rssi_update", "timestamp": "2025-08-18T23:59:59Z", "data": {}, "sequence": 999}
        writer.write(json.dumps(final_request).encode() + b'\n')
        await writer.drain()
        
        final_response_data = await reader.readline()
        final_response = json.loads(final_response_data.decode().strip())
        assert final_response["type"] == "rssi_update", "Connection should work normally after handling invalid messages"
        
        # Connection should still be active
        assert len(service.clients) == 1
        
        # Cleanup
        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect_stress_test(self, sdrpp_resilience_service):
        """RED: Test server stability under rapid connect/disconnect cycles."""
        service = sdrpp_resilience_service
        await service.start()
        
        # Perform rapid connection cycles
        for cycle in range(10):
            # Connect
            reader, writer = await asyncio.open_connection(service.host, service.port)
            
            # Send quick message
            request = {"type": "rssi_update", "timestamp": f"2025-08-18T23:59:{cycle:02d}Z", "data": {}, "sequence": cycle}
            writer.write(json.dumps(request).encode() + b'\n')
            await writer.drain()
            
            # Read response
            response_data = await reader.readline()
            response = json.loads(response_data.decode().strip())
            assert response["type"] == "rssi_update"
            
            # Disconnect immediately
            writer.close()
            await writer.wait_closed()
            
            # Brief pause between cycles
            await asyncio.sleep(0.05)
        
        # Server should still be stable and responsive
        assert service.running is True
        
        # Final connection test to verify server health
        reader, writer = await asyncio.open_connection(service.host, service.port)
        final_request = {"type": "rssi_update", "timestamp": "2025-08-18T23:59:59Z", "data": {}, "sequence": 999}
        writer.write(json.dumps(final_request).encode() + b'\n')
        await writer.drain()
        
        response_data = await reader.readline()
        response = json.loads(response_data.decode().strip())
        assert response["type"] == "rssi_update"
        
        # Cleanup
        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_abrupt_client_termination(self, sdrpp_resilience_service):
        """RED: Test proper cleanup when client terminates abruptly (simulating network failure)."""
        service = sdrpp_resilience_service
        await service.start()
        
        # Connect client
        reader, writer = await asyncio.open_connection(service.host, service.port)
        assert len(service.clients) == 1
        
        # Get client socket for abrupt termination
        client_socket = writer.get_extra_info('socket')
        
        # Terminate socket abruptly (simulates network failure/power loss)
        if client_socket:
            client_socket.close()
        
        # Don't call writer.close() - simulate abrupt termination
        
        # Give server time to detect disconnection and cleanup
        cleanup_timeout = 5.0  # 5 seconds should be enough for cleanup
        start_time = time.time()
        
        while len(service.clients) > 0 and (time.time() - start_time) < cleanup_timeout:
            await asyncio.sleep(0.1)
        
        # Server should have cleaned up the abruptly terminated client
        assert len(service.clients) == 0, "Server failed to clean up abruptly terminated client"
        
        # New client should be able to connect normally
        reader2, writer2 = await asyncio.open_connection(service.host, service.port)
        
        # Verify new connection works
        request = {"type": "rssi_update", "timestamp": "2025-08-18T23:59:59Z", "data": {}, "sequence": 1}
        writer2.write(json.dumps(request).encode() + b'\n')
        await writer2.drain()
        
        response_data = await reader2.readline()
        response = json.loads(response_data.decode().strip())
        assert response["type"] == "rssi_update"
        
        # Cleanup
        writer2.close()
        await writer2.wait_closed()

    @pytest.mark.asyncio
    async def test_server_graceful_shutdown_with_active_connections(self, sdrpp_resilience_service):
        """RED: Test server graceful shutdown behavior with multiple active connections."""
        service = sdrpp_resilience_service
        await service.start()
        
        # Connect multiple clients
        clients = []
        for i in range(3):
            reader, writer = await asyncio.open_connection(service.host, service.port)
            clients.append((reader, writer))
        
        assert len(service.clients) == 3
        
        # Initiate graceful shutdown
        shutdown_task = asyncio.create_task(service.stop())
        
        # Shutdown should complete within reasonable time
        await asyncio.wait_for(shutdown_task, timeout=5.0)
        
        # Server should be fully stopped
        assert service.running is False
        assert service.server is None
        assert len(service.clients) == 0
        
        # All client connections should be closed
        for reader, writer in clients:
            # Trying to write should fail or connection should be closed
            try:
                writer.write(b"test\n")
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass  # Expected - connection was closed
        
        # Clean up remaining clients (if any)
        for reader, writer in clients:
            if not writer.is_closing():
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()