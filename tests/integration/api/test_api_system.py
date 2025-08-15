"""Integration tests for system API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app
from src.backend.services.state_machine import SystemState


@pytest.fixture
def app():
    """Create test app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_state_machine():
    """Mock state machine for testing."""
    mock = MagicMock()
    mock.get_current_state = MagicMock(return_value=SystemState.IDLE)
    mock.get_state_string = MagicMock(return_value="IDLE")
    mock.transition_to = AsyncMock(return_value=True)
    mock.get_statistics = MagicMock(
        return_value={
            "total_transitions": 10,
            "time_in_states": {"IDLE": 100, "SEARCHING": 50},
            "transition_count": {"IDLE": 5, "SEARCHING": 5},
        }
    )
    mock.get_state_history = MagicMock(return_value=[])
    mock._is_running = True
    mock._homing_enabled = False
    return mock


@pytest.fixture
def mock_signal_processor():
    """Mock signal processor for testing."""
    mock = MagicMock()
    mock.get_current_rssi = MagicMock(return_value=-75.0)
    mock.get_current_snr = MagicMock(return_value=15.0)
    mock.get_noise_floor = MagicMock(return_value=-90.0)
    mock.is_running = True
    return mock


@pytest.fixture
def mock_mavlink_service():
    """Mock MAVLink service for testing."""
    mock = MagicMock()
    mock.is_connected = MagicMock(return_value=True)
    mock.get_telemetry = AsyncMock(
        return_value={
            "lat": 37.4419,
            "lon": -122.1430,
            "alt": 100.0,
            "heading": 90.0,
            "groundspeed": 5.0,
            "battery_voltage": 12.6,
            "battery_remaining": 85,
        }
    )
    return mock


class TestSystemStatusEndpoint:
    """Test /api/system/status endpoint."""

    def test_get_system_status(
        self, client, mock_state_machine, mock_signal_processor, mock_mavlink_service
    ):
        """Test getting system status."""
        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            with patch("src.backend.api.routes.system.signal_processor", mock_signal_processor):
                with patch("src.backend.api.routes.system.mavlink_service", mock_mavlink_service):
                    response = client.get("/api/system/status")

                    assert response.status_code == 200
                    data = response.json()

                    assert "state" in data
                    assert data["state"]["current"] == "IDLE"
                    assert data["state"]["is_running"] is True
                    assert data["state"]["homing_enabled"] is False

                    assert "signal" in data
                    assert data["signal"]["rssi"] == -75.0
                    assert data["signal"]["snr"] == 15.0
                    assert data["signal"]["noise_floor"] == -90.0

                    assert "mavlink" in data
                    assert data["mavlink"]["connected"] is True

    def test_get_system_status_without_services(self, client):
        """Test getting system status when services are not initialized."""
        with patch("src.backend.api.routes.system.state_machine", None):
            with patch("src.backend.api.routes.system.signal_processor", None):
                with patch("src.backend.api.routes.system.mavlink_service", None):
                    response = client.get("/api/system/status")

                    assert response.status_code == 200
                    data = response.json()

                    assert data["state"]["current"] == "UNKNOWN"
                    assert data["signal"]["rssi"] is None
                    assert data["mavlink"]["connected"] is False


class TestSystemStateEndpoints:
    """Test state transition endpoints."""

    def test_transition_state(self, client, mock_state_machine):
        """Test state transition."""
        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            response = client.post(
                "/api/system/state",
                json={"target_state": "SEARCHING", "reason": "Manual transition"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["current_state"] == "IDLE"

            mock_state_machine.transition_to.assert_called_once()

    def test_transition_state_invalid(self, client, mock_state_machine):
        """Test invalid state transition."""
        mock_state_machine.transition_to = AsyncMock(return_value=False)

        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            response = client.post(
                "/api/system/state", json={"target_state": "INVALID_STATE", "reason": "Test"}
            )

            # Should still return 200 but with success=False
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False

    def test_get_state_history(self, client, mock_state_machine):
        """Test getting state history."""
        mock_state_machine.get_state_history = MagicMock(
            return_value=[
                {
                    "from_state": "IDLE",
                    "to_state": "SEARCHING",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "reason": "Signal detected",
                }
            ]
        )

        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            response = client.get("/api/system/state-history")

            assert response.status_code == 200
            data = response.json()
            assert "history" in data
            assert len(data["history"]) == 1
            assert data["history"][0]["from_state"] == "IDLE"


class TestSystemControlEndpoints:
    """Test system control endpoints."""

    def test_emergency_stop(self, client, mock_state_machine):
        """Test emergency stop."""
        mock_state_machine.emergency_stop = AsyncMock(return_value=True)

        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            response = client.post(
                "/api/system/emergency-stop", json={"reason": "Test emergency stop"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            mock_state_machine.emergency_stop.assert_called_once()

    def test_enable_homing(self, client, mock_state_machine):
        """Test enabling homing."""
        mock_state_machine.enable_homing = MagicMock()
        mock_state_machine._current_state = SystemState.DETECTING

        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            response = client.post("/api/system/homing/enable")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            mock_state_machine.enable_homing.assert_called_once_with(True)

    def test_disable_homing(self, client, mock_state_machine):
        """Test disabling homing."""
        mock_state_machine.enable_homing = MagicMock()

        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            response = client.post("/api/system/homing/disable")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            mock_state_machine.enable_homing.assert_called_once_with(False)


class TestSystemStatisticsEndpoint:
    """Test system statistics endpoints."""

    def test_get_statistics(self, client, mock_state_machine):
        """Test getting system statistics."""
        with patch("src.backend.api.routes.system.state_machine", mock_state_machine):
            response = client.get("/api/system/statistics")

            assert response.status_code == 200
            data = response.json()
            assert "statistics" in data
            assert data["statistics"]["total_transitions"] == 10
