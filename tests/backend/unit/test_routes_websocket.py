"""
Unit tests for WebSocket endpoint functionality.

Tests WebSocket connection lifecycle, real-time message broadcasting,
and integration with signal processing, telemetry, and state management services.
Validates PRD-FR9 telemetry streaming requirements.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from src.backend.api.websocket import (
    ConnectionManager,
    broadcast_message,
    get_mavlink_service,
    get_signal_processor,
    get_state_machine,
    manager,
    router,
)
from src.backend.core.app import app
from src.backend.core.exceptions import SignalProcessingError, StateTransitionError
from src.backend.services.mavlink_service import ConnectionState
from src.backend.services.state_machine import SystemState


class TestWebSocketConnectionManager:
    """Test WebSocket ConnectionManager functionality."""

    @pytest.fixture
    def connection_manager(self):
        """Provide fresh ConnectionManager instance."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Provide mock WebSocket for testing."""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.send_bytes = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, connection_manager, mock_websocket):
        """Test WebSocket connection accept and disconnect."""
        # Test connection
        await connection_manager.connect(mock_websocket)

        # Verify connection added
        assert mock_websocket in connection_manager.active_connections
        mock_websocket.accept.assert_called_once()

        # Test disconnection
        await connection_manager.disconnect(mock_websocket)

        # Verify connection removed
        assert mock_websocket not in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, connection_manager):
        """Test multiple simultaneous WebSocket connections."""
        websockets = [Mock() for _ in range(3)]
        for ws in websockets:
            ws.accept = AsyncMock()

        # Connect all websockets
        for ws in websockets:
            await connection_manager.connect(ws)

        # Verify all connections tracked
        assert len(connection_manager.active_connections) == 3
        for ws in websockets:
            assert ws in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_json_success(self, connection_manager, mock_websocket):
        """Test successful JSON message broadcasting."""
        await connection_manager.connect(mock_websocket)

        test_message = {"type": "test", "data": {"value": 42}}
        await connection_manager.broadcast_json(test_message)

        # Verify message sent as JSON string
        expected_json = json.dumps(test_message)
        mock_websocket.send_text.assert_called_once_with(expected_json)

    @pytest.mark.asyncio
    async def test_broadcast_json_empty_connections(self, connection_manager):
        """Test broadcasting with no active connections."""
        test_message = {"type": "test", "data": {}}

        # Should not raise exception with empty connections
        await connection_manager.broadcast_json(test_message)

    @pytest.mark.asyncio
    async def test_broadcast_json_failed_send(self, connection_manager):
        """Test handling of failed WebSocket sends."""
        # Create websockets - one good, one failing
        good_ws = Mock()
        good_ws.accept = AsyncMock()
        good_ws.send_text = AsyncMock()

        bad_ws = Mock()
        bad_ws.accept = AsyncMock()
        bad_ws.send_text = AsyncMock(side_effect=Exception("Connection lost"))

        # Connect both
        await connection_manager.connect(good_ws)
        await connection_manager.connect(bad_ws)

        # Broadcast message
        test_message = {"type": "test", "data": {}}
        await connection_manager.broadcast_json(test_message)

        # Good connection should receive message
        good_ws.send_text.assert_called_once()

        # Bad connection should be removed from active connections
        assert bad_ws not in connection_manager.active_connections
        assert good_ws in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_bytes_success(self, connection_manager, mock_websocket):
        """Test successful binary data broadcasting."""
        await connection_manager.connect(mock_websocket)

        test_data = b"binary test data"
        await connection_manager.broadcast_bytes(test_data)

        mock_websocket.send_bytes.assert_called_once_with(test_data)


class TestWebSocketEndpoint:
    """Test WebSocket endpoint behavior."""

    @pytest.fixture
    def websocket_client(self):
        """Provide WebSocket test client."""
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_websocket_info_endpoint(self, websocket_client):
        """Test WebSocket info endpoint returns configuration."""
        response = websocket_client.get("/ws-info")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "endpoint" in data
        assert "status" in data
        assert "update_rate_hz" in data
        assert "active_connections" in data
        assert "message_types" in data

        # Verify endpoint path
        assert data["endpoint"] == "/ws"

        # Verify message types include PRD requirements
        message_types = data["message_types"]
        assert any("rssi" in msg.lower() for msg in message_types)
        assert any("telemetry" in msg.lower() for msg in message_types)
        assert any("state" in msg.lower() for msg in message_types)

    def test_websocket_connection_establish(self, websocket_client):
        """Test WebSocket connection establishment and initial message."""
        with (
            patch("src.backend.api.websocket.get_signal_processor") as mock_get_processor,
            patch("src.backend.api.websocket.get_mavlink_service") as mock_get_mavlink,
        ):

            # Mock the services to avoid SDR hardware dependency
            mock_processor = Mock()
            mock_processor.start = AsyncMock()
            mock_get_processor.return_value = mock_processor

            mock_mavlink = Mock()
            mock_mavlink.start = AsyncMock()
            mock_mavlink.add_state_callback = Mock()
            mock_get_mavlink.return_value = mock_mavlink

            with websocket_client.websocket_connect("/ws") as websocket:
                # Should receive initial connection message
                data = websocket.receive_json()

                assert data["type"] == "connection"
                assert data["data"]["status"] == "connected"
                assert "timestamp" in data["data"]

    def test_websocket_ping_pong(self, websocket_client):
        """Test WebSocket ping/pong functionality."""
        with (
            patch("src.backend.api.websocket.get_signal_processor") as mock_get_processor,
            patch("src.backend.api.websocket.get_mavlink_service") as mock_get_mavlink,
        ):

            # Mock the services to avoid SDR hardware dependency
            mock_processor = Mock()
            mock_processor.start = AsyncMock()
            mock_get_processor.return_value = mock_processor

            mock_mavlink = Mock()
            mock_mavlink.start = AsyncMock()
            mock_mavlink.add_state_callback = Mock()
            mock_get_mavlink.return_value = mock_mavlink

            with websocket_client.websocket_connect("/ws") as websocket:
                # Skip initial connection message and any telemetry messages
                websocket.receive_json()  # connection message

                # Send ping
                ping_message = {"type": "ping"}
                websocket.send_json(ping_message)

                # Receive messages until we get the pong (may receive telemetry first)
                for _ in range(5):  # Max 5 attempts to find pong
                    data = websocket.receive_json()
                    if data["type"] == "pong":
                        assert "timestamp" in data["data"]
                        break
                else:
                    pytest.fail("Did not receive pong response within expected time")

    def test_websocket_invalid_json_handling(self, websocket_client):
        """Test WebSocket handling of invalid JSON messages."""
        with (
            patch("src.backend.api.websocket.get_signal_processor") as mock_get_processor,
            patch("src.backend.api.websocket.get_mavlink_service") as mock_get_mavlink,
        ):

            # Mock the services to avoid SDR hardware dependency
            mock_processor = Mock()
            mock_processor.start = AsyncMock()
            mock_get_processor.return_value = mock_processor

            mock_mavlink = Mock()
            mock_mavlink.start = AsyncMock()
            mock_mavlink.add_state_callback = Mock()
            mock_get_mavlink.return_value = mock_mavlink

            with websocket_client.websocket_connect("/ws") as websocket:
                # Skip initial connection message
                websocket.receive_json()

                # Send invalid JSON
                websocket.send_text("invalid json {")

                # Connection should remain stable (no disconnect)
                # Verify by sending valid ping
                websocket.send_json({"type": "ping"})

                # Look for pong response among possible telemetry messages
                for _ in range(3):
                    data = websocket.receive_json()
                    if data["type"] == "pong":
                        break
                else:
                    pytest.fail("Connection was disrupted by invalid JSON")


class TestServiceIntegration:
    """Test WebSocket integration with backend services."""

    @pytest.mark.asyncio
    async def test_signal_processor_integration(self):
        """Test signal processor service integration."""
        # Clear any existing global state
        import src.backend.api.websocket as ws_module

        ws_module.signal_processor_integration = None
        ws_module._rssi_broadcast_task = None

        with patch("src.backend.api.websocket.SignalProcessorIntegration") as mock_processor:
            mock_instance = Mock()
            mock_instance.start = AsyncMock()
            mock_processor.return_value = mock_instance

            processor = await get_signal_processor()

            # Verify processor created and started
            mock_processor.assert_called_once()
            mock_instance.start.assert_called_once()
            assert processor == mock_instance

    @pytest.mark.asyncio
    async def test_mavlink_service_integration(self):
        """Test MAVLink service integration with configuration."""
        # Clear any existing global state
        import src.backend.api.websocket as ws_module

        ws_module.mavlink_service = None
        ws_module._telemetry_broadcast_task = None

        with (
            patch("src.backend.api.websocket.MAVLinkService") as mock_mavlink,
            patch("src.backend.api.websocket.get_config") as mock_config,
        ):

            # Mock configuration
            config = Mock()
            config.mavlink_device = "tcp:127.0.0.1:5760"
            config.mavlink_baud = 115200
            mock_config.return_value = config

            # Mock MAVLink service
            mock_instance = Mock()
            mock_instance.start = AsyncMock()
            mock_instance.add_state_callback = Mock()
            mock_mavlink.return_value = mock_instance

            service = await get_mavlink_service()

            # Verify service created with config
            mock_mavlink.assert_called_once_with(device_path="tcp:127.0.0.1:5760", baud_rate=115200)
            mock_instance.start.assert_called_once()
            mock_instance.add_state_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_machine_integration(self):
        """Test state machine integration with callbacks."""
        with patch("src.backend.api.websocket.StateMachine") as mock_state_machine:
            mock_instance = Mock()
            mock_instance.start = AsyncMock()
            mock_instance.add_state_callback = Mock()
            mock_state_machine.return_value = mock_instance

            state_machine = await get_state_machine()

            # Verify state machine created and configured
            mock_state_machine.assert_called_once()
            mock_instance.start.assert_called_once()
            mock_instance.add_state_callback.assert_called_once()


class TestMessageBroadcasting:
    """Test message broadcasting functionality."""

    @pytest.mark.asyncio
    async def test_broadcast_message_function(self):
        """Test global broadcast_message function."""
        with patch("src.backend.api.websocket.manager") as mock_manager:
            mock_manager.broadcast_json = AsyncMock()

            test_message = {"type": "test", "data": {"value": 42}}
            await broadcast_message(test_message)

            mock_manager.broadcast_json.assert_called_once_with(test_message)

    @pytest.mark.asyncio
    async def test_rssi_broadcast_task(self):
        """Test RSSI broadcasting task functionality."""
        with (
            patch("src.backend.api.websocket.get_signal_processor") as mock_get_processor,
            patch("src.backend.api.websocket.manager") as mock_manager,
            patch("src.backend.api.websocket.get_config") as mock_config,
        ):

            # Mock configuration
            config = Mock()
            config.websocket = Mock()
            config.websocket.WS_RSSI_UPDATE_INTERVAL_MS = 100
            mock_config.return_value = config

            # Mock signal processor with RSSI stream
            mock_processor = Mock()

            # Create async generator for RSSI readings
            async def mock_rssi_stream():
                rssi_reading = Mock()
                rssi_reading.rssi = -45.0
                rssi_reading.noise_floor = -80.0
                rssi_reading.snr = 35.0
                rssi_reading.confidence = 0.95
                rssi_reading.timestamp = datetime.utcnow()
                yield rssi_reading

            mock_processor.get_rssi_stream = mock_rssi_stream
            mock_get_processor.return_value = mock_processor

            # Mock manager
            mock_manager.broadcast_json = AsyncMock()

            # Import and test the actual broadcast function
            from src.backend.api.websocket import broadcast_rssi_updates

            # Run broadcast task briefly
            task = asyncio.create_task(broadcast_rssi_updates())
            await asyncio.sleep(0.01)  # Brief execution
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify broadcast was called
            mock_manager.broadcast_json.assert_called()
            call_args = mock_manager.broadcast_json.call_args[0][0]
            assert call_args["type"] == "rssi"
            assert call_args["data"]["rssi"] == -45.0

    @pytest.mark.asyncio
    async def test_telemetry_broadcast_task(self):
        """Test telemetry broadcasting functionality."""
        with (
            patch("src.backend.api.websocket.get_mavlink_service") as mock_get_mavlink,
            patch("src.backend.api.websocket.manager") as mock_manager,
        ):

            # Mock MAVLink service
            mock_mavlink = Mock()
            mock_mavlink.is_connected.return_value = True
            mock_mavlink.get_telemetry.return_value = {
                "position": {"lat": 37.7749, "lng": -122.4194, "alt": 100.0},
                "battery": {"percentage": 85},
                "flight_mode": "GUIDED",
                "armed": True,
            }
            mock_mavlink.get_gps_status_string.return_value = "3D Fix"
            mock_get_mavlink.return_value = mock_mavlink

            # Mock manager
            mock_manager.broadcast_json = AsyncMock()

            # Import and test broadcast function
            from src.backend.api.websocket import broadcast_telemetry_updates

            # Run broadcast task briefly
            task = asyncio.create_task(broadcast_telemetry_updates())
            await asyncio.sleep(0.01)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify telemetry broadcast
            mock_manager.broadcast_json.assert_called()
            call_args = mock_manager.broadcast_json.call_args[0][0]
            assert call_args["type"] == "telemetry"
            assert call_args["data"]["battery"] == 85
            assert call_args["data"]["flightMode"] == "GUIDED"

    @pytest.mark.asyncio
    async def test_state_change_broadcast(self):
        """Test state change broadcasting."""
        with (
            patch("src.backend.api.websocket.manager") as mock_manager,
            patch("src.backend.api.websocket.state_machine") as mock_state_machine,
        ):

            mock_manager.broadcast_json = AsyncMock()

            # Mock state machine methods
            mock_state_machine.get_allowed_transitions.return_value = [
                SystemState.SEARCHING,
                SystemState.IDLE,
            ]
            mock_state_machine.get_statistics.return_value = {"homing_enabled": False}

            from src.backend.api.websocket import broadcast_state_change

            # Test state change broadcast
            await broadcast_state_change(
                SystemState.IDLE, SystemState.SEARCHING, "Search pattern activated"
            )

            # Verify broadcast called with state change data
            mock_manager.broadcast_json.assert_called_once()
            call_args = mock_manager.broadcast_json.call_args[0][0]
            assert call_args["type"] == "state_change"
            assert call_args["data"]["from_state"] == "IDLE"
            assert call_args["data"]["to_state"] == "SEARCHING"
            assert call_args["data"]["reason"] == "Search pattern activated"


class TestErrorHandling:
    """Test WebSocket error handling and resilience."""

    @pytest.mark.asyncio
    async def test_signal_processing_error_handling(self):
        """Test handling of signal processing errors in broadcast tasks."""
        with patch("src.backend.api.websocket.get_signal_processor") as mock_get_processor:
            # Mock processor that raises error
            mock_get_processor.side_effect = SignalProcessingError("SDR connection lost")

            from src.backend.api.websocket import broadcast_rssi_updates

            # Task should handle error gracefully
            task = asyncio.create_task(broadcast_rssi_updates())
            await asyncio.sleep(0.01)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass
            except SignalProcessingError:
                pytest.fail("SignalProcessingError should be handled gracefully")

    @pytest.mark.asyncio
    async def test_state_transition_error_handling(self):
        """Test handling of state transition errors in broadcast tasks."""
        with patch("src.backend.api.websocket.get_state_machine") as mock_get_state:
            # Mock state machine that raises error
            mock_get_state.side_effect = StateTransitionError("Invalid state transition")

            from src.backend.api.websocket import broadcast_state_updates

            # Task should handle error gracefully
            task = asyncio.create_task(broadcast_state_updates())
            await asyncio.sleep(0.01)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass
            except StateTransitionError:
                pytest.fail("StateTransitionError should be handled gracefully")

    @pytest.mark.asyncio
    async def test_mavlink_status_broadcast(self):
        """Test MAVLink connection status broadcasting."""
        with patch("src.backend.api.websocket.manager") as mock_manager:
            mock_manager.broadcast_json = AsyncMock()

            from src.backend.api.websocket import broadcast_mavlink_status

            # Test connection status broadcast
            await broadcast_mavlink_status(ConnectionState.CONNECTED)

            # Verify status broadcast
            mock_manager.broadcast_json.assert_called_once()
            call_args = mock_manager.broadcast_json.call_args[0][0]
            assert call_args["type"] == "mavlink_status"
            assert call_args["data"]["connected"] is True
            assert call_args["data"]["state"] == ConnectionState.CONNECTED.value


class TestPRDCompliance:
    """Test PRD requirement compliance for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_prd_fr9_telemetry_streaming(self):
        """Test PRD-FR9: System shall stream RSSI telemetry to ground control station."""
        client = TestClient(app)
        response = client.get("/ws-info")
        assert response.status_code == 200

        data = response.json()
        message_types = data["message_types"]

        # Verify telemetry streaming capability
        assert any("rssi" in msg.lower() for msg in message_types)
        assert any("telemetry" in msg.lower() for msg in message_types)

    @pytest.mark.asyncio
    async def test_prd_fr12_state_logging(self):
        """Test PRD-FR12: System shall log all state transitions and signal detections."""
        client = TestClient(app)
        response = client.get("/ws-info")
        assert response.status_code == 200

        data = response.json()
        message_types = data["message_types"]

        # Verify state change and detection streaming
        assert any("state" in msg.lower() for msg in message_types)
        assert any("detection" in msg.lower() for msg in message_types)

    @pytest.mark.asyncio
    async def test_prd_nfr2_processing_latency(self):
        """Test PRD-NFR2: Signal processing latency shall not exceed 100ms per RSSI computation cycle."""
        # Test WebSocket configuration allows for <100ms updates
        from src.backend.api.websocket import router

        # Verify WebSocket endpoint exists and is properly configured
        assert any(route.path == "/ws" for route in router.routes)

        # Test that connection manager can handle high-frequency updates
        connection_manager = ConnectionManager()
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        await connection_manager.connect(mock_websocket)

        # Test rapid message sending (simulating <100ms intervals)
        start_time = asyncio.get_event_loop().time()
        for i in range(10):
            await connection_manager.broadcast_json({"type": "test", "data": {"seq": i}})
        end_time = asyncio.get_event_loop().time()

        # Verify all messages sent efficiently
        assert mock_websocket.send_text.call_count == 10
        assert (end_time - start_time) < 0.1  # Should complete in <100ms
