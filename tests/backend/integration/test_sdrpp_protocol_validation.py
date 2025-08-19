"""
JSON Protocol Validation Tests for SDR++ Communication

Tests verify authentic bidirectional JSON message protocol with all message types.
Following TDD methodology - testing protocol compliance and error handling.

PRD References:
- NFR1: Communication reliability with message validation
- NFR2: Signal processing latency (<100ms) with protocol overhead
- FR9: Enhanced telemetry streaming with dual-SDR coordination

Protocol Message Types:
- rssi_update: RSSI data streaming from drone to ground
- freq_control: Frequency control commands from ground to drone
- heartbeat: Connection monitoring messages
- error: Error responses for invalid messages
"""

import asyncio
import json
import socket
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService


class TestSDRPPProtocolValidation:
    """Protocol validation tests for SDR++ JSON message format."""

    @pytest.fixture
    def unused_tcp_port(self):
        """Provide unused TCP port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @pytest.fixture
    async def protocol_service(self, unused_tcp_port):
        """Provide SDRPPBridgeService instance for protocol testing."""
        service = SDRPPBridgeService()
        service.port = unused_tcp_port

        # Mock signal processor with realistic values
        mock_processor = MagicMock()  # Remove spec to allow dynamic attributes
        mock_processor.get_current_rssi.return_value = -67.8
        mock_processor.set_frequency.return_value = None  # Add set_frequency method
        service.set_signal_processor(mock_processor)

        await service.start()
        yield service
        await service.stop()

    async def send_json_message(self, writer, message_dict):
        """Helper to send JSON message with newline delimiter."""
        message_json = json.dumps(message_dict) + "\n"
        writer.write(message_json.encode())
        await writer.drain()

    async def receive_json_message(self, reader):
        """Helper to receive and parse JSON message."""
        line = await reader.readline()
        return json.loads(line.decode().strip())

    @pytest.mark.asyncio
    async def test_rssi_update_request_protocol(self, protocol_service):
        """RED: Test RSSI update request message protocol compliance."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send valid RSSI update request
        request = {
            "type": "rssi_update",
            "timestamp": "2025-08-18T22:55:00Z",
            "data": {},
            "sequence": 100,
        }

        await self.send_json_message(writer, request)

        # Receive response
        response = await self.receive_json_message(reader)

        # Verify protocol compliance
        assert response["type"] == "rssi_update"
        assert "timestamp" in response
        assert "data" in response
        assert "sequence" in response

        # Verify data structure
        assert "rssi" in response["data"]
        assert "frequency" in response["data"]
        assert isinstance(response["data"]["rssi"], float)
        assert isinstance(response["data"]["frequency"], int)

        # Verify realistic values
        assert -120.0 <= response["data"]["rssi"] <= -20.0  # Typical RSSI range
        assert response["data"]["frequency"] > 0  # Valid frequency

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_frequency_control_message_protocol(self, protocol_service):
        """RED: Test frequency control message protocol validation."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send valid frequency control command
        freq_control = {
            "type": "freq_control",
            "timestamp": "2025-08-18T22:55:30Z",
            "data": {"frequency": 2437000000},  # 2.437 GHz WiFi frequency
            "sequence": 101,
        }

        await self.send_json_message(writer, freq_control)

        # Receive response
        response = await self.receive_json_message(reader)

        # Verify protocol compliance
        assert response["type"] == "freq_control_response"
        assert "timestamp" in response
        assert "data" in response
        assert "sequence" in response

        # Verify successful frequency control response
        assert "status" in response["data"]
        assert response["data"]["status"] == "success"
        assert "frequency" in response["data"]
        assert response["data"]["frequency"] == 2437000000

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_heartbeat_message_protocol(self, protocol_service):
        """RED: Test heartbeat message protocol handling."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send heartbeat message
        heartbeat = {
            "type": "heartbeat",
            "timestamp": "2025-08-18T22:56:00Z",
            "data": {"client_id": "sdrpp_plugin_001"},
            "sequence": 102,
        }

        await self.send_json_message(writer, heartbeat)

        # Receive response
        response = await self.receive_json_message(reader)

        # Verify heartbeat acknowledgment
        assert response["type"] == "heartbeat_ack"
        assert "timestamp" in response
        assert "data" in response
        assert "sequence" in response
        assert response["data"]["status"] == "received"

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_invalid_json_error_handling(self, protocol_service):
        """RED: Test protocol error handling for malformed JSON."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send invalid JSON
        invalid_json = '{"type": "rssi_update", "invalid": json,}\n'
        writer.write(invalid_json.encode())
        await writer.drain()

        # Receive error response
        response = await self.receive_json_message(reader)

        # Verify error protocol compliance
        assert response["type"] == "error"
        assert "timestamp" in response
        assert "data" in response
        assert "sequence" in response
        assert "error" in response["data"]
        assert "Invalid JSON message format" in response["data"]["error"]

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_unknown_message_type_error(self, protocol_service):
        """RED: Test protocol error handling for unknown message types."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send unknown message type
        unknown_message = {
            "type": "unknown_command",
            "timestamp": "2025-08-18T22:57:00Z",
            "data": {},
            "sequence": 103,
        }

        await self.send_json_message(writer, unknown_message)

        # Receive error response
        response = await self.receive_json_message(reader)

        # Verify error protocol compliance
        assert response["type"] == "error"
        assert "timestamp" in response
        assert "data" in response
        assert "sequence" in response
        assert "error" in response["data"]
        assert "Unknown message type: unknown_command" in response["data"]["error"]

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_frequency_validation_error_handling(self, protocol_service):
        """RED: Test frequency range validation error handling."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send invalid frequency (outside PRD-FR1 range: 850 MHz - 6.5 GHz)
        invalid_freq = {
            "type": "freq_control",
            "timestamp": "2025-08-18T22:57:30Z",
            "data": {"frequency": 100000000},  # 100 MHz - below minimum
            "sequence": 104,
        }

        await self.send_json_message(writer, invalid_freq)

        # Receive error response
        response = await self.receive_json_message(reader)

        # Verify error response for invalid frequency
        assert response["type"] == "error"
        assert "data" in response
        assert response["data"]["error"] == "frequency_out_of_range"
        assert "message" in response["data"]
        assert "100.0 MHz" in response["data"]["message"]

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_message_sequence_tracking(self, protocol_service):
        """RED: Test message sequence number tracking."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send multiple messages and track sequence numbers
        sequences = []
        for i in range(3):
            request = {
                "type": "rssi_update",
                "timestamp": f"2025-08-18T22:58:{i:02d}Z",
                "data": {},
                "sequence": 200 + i,
            }

            await self.send_json_message(writer, request)
            response = await self.receive_json_message(reader)
            sequences.append(response["sequence"])

        # Verify sequence numbers are unique and increasing
        assert len(set(sequences)) == 3  # All unique
        assert sequences == sorted(sequences)  # Increasing order

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_timestamp_format_validation(self, protocol_service):
        """RED: Test timestamp format in protocol messages."""
        reader, writer = await asyncio.open_connection(protocol_service.host, protocol_service.port)

        # Send message and verify timestamp format in response
        request = {
            "type": "rssi_update",
            "timestamp": "2025-08-18T22:59:00Z",
            "data": {},
            "sequence": 300,
        }

        await self.send_json_message(writer, request)
        response = await self.receive_json_message(reader)

        # Verify timestamp is valid ISO 8601 format
        timestamp_str = response["timestamp"]
        assert "T" in timestamp_str
        assert timestamp_str.endswith("Z") or "+" in timestamp_str

        # Should be parseable as datetime
        parsed_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert parsed_time is not None

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_concurrent_protocol_message_handling(self, protocol_service):
        """RED: Test protocol handling under concurrent message load."""
        # Connect multiple clients
        clients = []
        for i in range(3):
            reader, writer = await asyncio.open_connection(
                protocol_service.host, protocol_service.port
            )
            clients.append((reader, writer))

        # Send concurrent messages from all clients
        async def send_and_verify(client_id, reader, writer):
            request = {
                "type": "rssi_update",
                "timestamp": f"2025-08-18T23:00:{client_id:02d}Z",
                "data": {},
                "sequence": 400 + client_id,
            }

            await self.send_json_message(writer, request)
            response = await self.receive_json_message(reader)

            # Verify protocol compliance
            assert response["type"] == "rssi_update"
            assert "data" in response
            assert "rssi" in response["data"]

            return response

        # Execute concurrent requests
        tasks = [send_and_verify(i, reader, writer) for i, (reader, writer) in enumerate(clients)]

        responses = await asyncio.gather(*tasks)

        # Verify all responses valid
        assert len(responses) == 3
        for response in responses:
            assert response["type"] == "rssi_update"
            assert -120.0 <= response["data"]["rssi"] <= -20.0

        # Cleanup
        for reader, writer in clients:
            writer.close()
            await writer.wait_closed()
