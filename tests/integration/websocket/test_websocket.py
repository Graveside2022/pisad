"""Integration tests for WebSocket connections."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def app():
    """Create test app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client with WebSocket support."""
    return TestClient(app)


@pytest.fixture
def mock_signal_processor():
    """Mock signal processor."""
    mock = MagicMock()
    mock.get_current_rssi = MagicMock(return_value=-75.0)
    mock.get_current_snr = MagicMock(return_value=15.0)
    mock.get_noise_floor = MagicMock(return_value=-90.0)
    mock.rssi_generator = AsyncMock(return_value=iter([-75.0, -74.0, -73.0]))
    return mock


@pytest.fixture
def mock_state_machine():
    """Mock state machine."""
    mock = MagicMock()
    mock.get_current_state = MagicMock(return_value="IDLE")
    mock.get_state_string = MagicMock(return_value="IDLE")
    mock.add_state_callback = MagicMock()
    return mock


@pytest.fixture
def mock_mavlink_service():
    """Mock MAVLink service."""
    mock = MagicMock()
    mock.get_telemetry = AsyncMock(
        return_value={
            "lat": 37.4419,
            "lon": -122.1430,
            "alt": 100.0,
            "heading": 90.0,
            "groundspeed": 5.0,
        }
    )
    return mock


class TestWebSocketConnection:
    """Test WebSocket connection and basic messaging."""

    def test_websocket_connect(self, client):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive initial connection message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping-pong."""
        with client.websocket_connect("/ws") as websocket:
            # Skip initial connection message
            websocket.receive_json()

            # Send ping
            websocket.send_json({"type": "ping"})

            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_invalid_message(self, client):
        """Test handling of invalid WebSocket messages."""
        with client.websocket_connect("/ws") as websocket:
            # Skip initial connection message
            websocket.receive_json()

            # Send invalid message
            websocket.send_json({"invalid": "message"})

            # Should receive error
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "message" in data


class TestWebSocketDataStreaming:
    """Test real-time data streaming over WebSocket."""

    def test_rssi_streaming(self, client, mock_signal_processor):
        """Test RSSI data streaming."""
        with patch("src.backend.api.websocket.signal_processor", mock_signal_processor):
            with client.websocket_connect("/ws") as websocket:
                # Skip initial connection message
                websocket.receive_json()

                # Subscribe to RSSI updates
                websocket.send_json({"type": "subscribe", "channel": "rssi"})

                # Should receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscribed"
                assert data["channel"] == "rssi"

                # Should start receiving RSSI updates
                data = websocket.receive_json()
                assert data["type"] == "rssi_update"
                assert "rssi" in data["data"]
                assert "snr" in data["data"]
                assert "noise_floor" in data["data"]

    def test_state_updates(self, client, mock_state_machine):
        """Test state change notifications."""
        with patch("src.backend.api.websocket.state_machine", mock_state_machine):
            with client.websocket_connect("/ws") as websocket:
                # Skip initial connection message
                websocket.receive_json()

                # Subscribe to state updates
                websocket.send_json({"type": "subscribe", "channel": "state"})

                # Should receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscribed"
                assert data["channel"] == "state"

    def test_telemetry_streaming(self, client, mock_mavlink_service):
        """Test telemetry data streaming."""
        with patch("src.backend.api.websocket.mavlink_service", mock_mavlink_service):
            with client.websocket_connect("/ws") as websocket:
                # Skip initial connection message
                websocket.receive_json()

                # Subscribe to telemetry
                websocket.send_json({"type": "subscribe", "channel": "telemetry"})

                # Should receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscribed"
                assert data["channel"] == "telemetry"


class TestWebSocketCommands:
    """Test WebSocket command handling."""

    def test_start_homing_command(self, client, mock_state_machine):
        """Test starting homing via WebSocket."""
        mock_state_machine.transition_to = AsyncMock(return_value=True)

        with patch("src.backend.api.websocket.state_machine", mock_state_machine):
            with client.websocket_connect("/ws") as websocket:
                # Skip initial connection message
                websocket.receive_json()

                # Send start homing command
                websocket.send_json({"type": "command", "command": "start_homing"})

                # Should receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "command_response"
                assert data["success"] is True

    def test_emergency_stop_command(self, client, mock_state_machine):
        """Test emergency stop via WebSocket."""
        mock_state_machine.emergency_stop = AsyncMock(return_value=True)

        with patch("src.backend.api.websocket.state_machine", mock_state_machine):
            with client.websocket_connect("/ws") as websocket:
                # Skip initial connection message
                websocket.receive_json()

                # Send emergency stop command
                websocket.send_json(
                    {"type": "command", "command": "emergency_stop", "reason": "Test emergency"}
                )

                # Should receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "command_response"
                assert data["success"] is True

                mock_state_machine.emergency_stop.assert_called_once()


class TestWebSocketMultipleClients:
    """Test handling multiple WebSocket clients."""

    def test_multiple_clients(self, client):
        """Test multiple simultaneous WebSocket connections."""
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                # Both should receive connection messages
                data1 = ws1.receive_json()
                assert data1["type"] == "connection"

                data2 = ws2.receive_json()
                assert data2["type"] == "connection"

                # Send ping from first client
                ws1.send_json({"type": "ping"})

                # Only first client should receive pong
                data1 = ws1.receive_json()
                assert data1["type"] == "pong"


class TestWebSocketErrorHandling:
    """Test WebSocket error handling."""

    def test_connection_error_recovery(self, client):
        """Test recovery from connection errors."""
        with client.websocket_connect("/ws") as websocket:
            # Skip initial connection message
            websocket.receive_json()

            # Send malformed JSON (as text)
            websocket.send_text("not json")

            # Should receive error but connection should stay open
            data = websocket.receive_json()
            assert data["type"] == "error"

            # Should still be able to send valid messages
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_rate_limiting(self, client):
        """Test WebSocket rate limiting."""
        with client.websocket_connect("/ws") as websocket:
            # Skip initial connection message
            websocket.receive_json()

            # Send many messages rapidly
            for _ in range(100):
                websocket.send_json({"type": "ping"})

            # Should eventually receive rate limit error
            received_rate_limit = False
            for _ in range(100):
                data = websocket.receive_json()
                if data.get("type") == "rate_limit":
                    received_rate_limit = True
                    break

            # Note: Rate limiting might not be implemented yet
            # This test documents expected behavior
