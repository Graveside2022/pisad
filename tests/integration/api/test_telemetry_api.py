"""Unit tests for telemetry API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.routes.telemetry import get_mavlink_service, router
from src.backend.services.mavlink_service import MAVLinkService

# Create test app
app = FastAPI()
app.include_router(router)


# Mock MAVLink service
@pytest.fixture
def mock_mavlink_service():
    """Create a mock MAVLink service."""
    service = MagicMock(spec=MAVLinkService)
    service.is_connected.return_value = True
    service._rssi_value = -75.0
    service.get_telemetry_config.return_value = {
        "rssi_rate_hz": 2.0,
        "health_interval_seconds": 10,
        "detection_throttle_ms": 500,
    }
    service.send_named_value_float.return_value = True
    service.send_statustext.return_value = True
    return service


@pytest.fixture
def client(mock_mavlink_service):
    """Create test client with mocked dependencies."""
    app.dependency_overrides[get_mavlink_service] = lambda: mock_mavlink_service
    return TestClient(app)


class TestTelemetryConfig:
    """Test telemetry configuration endpoints."""

    def test_get_telemetry_config(self, client):
        """Test getting telemetry configuration."""
        response = client.get("/api/telemetry/config")
        assert response.status_code == 200

        data = response.json()
        assert data["rssi_rate_hz"] == 2.0
        assert data["health_interval_seconds"] == 10
        assert data["detection_throttle_ms"] == 500
        assert "statustext_severity" in data
        assert "max_bandwidth_kbps" in data

    def test_update_telemetry_config(self, client, mock_mavlink_service):
        """Test updating telemetry configuration."""
        update_data = {
            "rssi_rate_hz": 1.0,
            "health_interval_seconds": 20,
        }

        response = client.put("/api/telemetry/config", json=update_data)
        assert response.status_code == 200

        # Verify update was called
        mock_mavlink_service.update_telemetry_config.assert_called_once()
        call_args = mock_mavlink_service.update_telemetry_config.call_args[0][0]
        assert call_args["rssi_rate_hz"] == 1.0
        assert call_args["health_interval_seconds"] == 20

    def test_update_telemetry_config_partial(self, client, mock_mavlink_service):
        """Test partial update of telemetry configuration."""
        update_data = {
            "rssi_rate_hz": 5.0,
        }

        response = client.put("/api/telemetry/config", json=update_data)
        assert response.status_code == 200

        # Verify only specified field was updated
        call_args = mock_mavlink_service.update_telemetry_config.call_args[0][0]
        assert call_args["rssi_rate_hz"] == 5.0
        assert "health_interval_seconds" not in call_args

    def test_update_telemetry_config_validation(self, client):
        """Test configuration validation."""
        # Test rate too high
        update_data = {"rssi_rate_hz": 100.0}
        response = client.put("/api/telemetry/config", json=update_data)
        assert response.status_code == 422  # Validation error

        # Test rate too low
        update_data = {"rssi_rate_hz": 0.01}
        response = client.put("/api/telemetry/config", json=update_data)
        assert response.status_code == 422

        # Test invalid health interval
        update_data = {"health_interval_seconds": 100}
        response = client.put("/api/telemetry/config", json=update_data)
        assert response.status_code == 422


class TestTelemetryStatus:
    """Test telemetry status endpoint."""

    def test_get_telemetry_status(self, client, mock_mavlink_service):
        """Test getting telemetry status."""
        response = client.get("/api/telemetry/status")
        assert response.status_code == 200

        data = response.json()
        assert data["connected"] is True
        assert data["current_rssi"] == -75.0
        assert "bandwidth_usage_kbps" in data
        assert "config" in data

        # Check bandwidth calculation is reasonable
        assert 0 < data["bandwidth_usage_kbps"] < 10.0

    def test_get_telemetry_status_disconnected(self, client, mock_mavlink_service):
        """Test status when MAVLink is disconnected."""
        mock_mavlink_service.is_connected.return_value = False

        response = client.get("/api/telemetry/status")
        assert response.status_code == 200

        data = response.json()
        assert data["connected"] is False


class TestTelemetryTesting:
    """Test telemetry testing endpoints."""

    def test_test_rssi_telemetry(self, client, mock_mavlink_service):
        """Test sending test RSSI value."""
        response = client.post("/api/telemetry/test/rssi", json={"rssi": -80.0})
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "-80" in data["message"]

        # Verify methods were called
        mock_mavlink_service.update_rssi_value.assert_called_with(-80.0)
        mock_mavlink_service.send_named_value_float.assert_called_with("PISAD_RSSI", -80.0)

    def test_test_rssi_validation(self, client):
        """Test RSSI value validation."""
        # Too high
        response = client.post("/api/telemetry/test/rssi", json={"rssi": 10.0})
        assert response.status_code == 422

        # Too low
        response = client.post("/api/telemetry/test/rssi", json={"rssi": -150.0})
        assert response.status_code == 422

    def test_test_rssi_disconnected(self, client, mock_mavlink_service):
        """Test RSSI test when disconnected."""
        mock_mavlink_service.is_connected.return_value = False

        response = client.post("/api/telemetry/test/rssi", json={"rssi": -80.0})
        assert response.status_code == 503
        assert "not connected" in response.json()["detail"]

    def test_test_status_message(self, client, mock_mavlink_service):
        """Test sending test status message."""
        response = client.post(
            "/api/telemetry/test/status", json={"message": "Test message", "severity": 6}
        )
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "Test message" in data["message"]

        # Verify statustext was sent
        mock_mavlink_service.send_statustext.assert_called_once()
        call_args = mock_mavlink_service.send_statustext.call_args[0]
        assert "PISAD: Test message" in call_args[0]
        assert call_args[1] == 6

    def test_test_status_message_validation(self, client):
        """Test status message validation."""
        # Message too long
        long_message = "x" * 50  # Too long with PISAD prefix
        response = client.post("/api/telemetry/test/status", json={"message": long_message})
        assert response.status_code == 422

        # Invalid severity
        response = client.post(
            "/api/telemetry/test/status", json={"message": "Test", "severity": 10}
        )
        assert response.status_code == 422

    def test_test_status_disconnected(self, client, mock_mavlink_service):
        """Test status message when disconnected."""
        mock_mavlink_service.is_connected.return_value = False

        response = client.post("/api/telemetry/test/status", json={"message": "Test"})
        assert response.status_code == 503


class TestErrorHandling:
    """Test error handling in telemetry API."""

    def test_config_exception_handling(self, client, mock_mavlink_service):
        """Test exception handling in config endpoint."""
        mock_mavlink_service.get_telemetry_config.side_effect = Exception("Test error")

        response = client.get("/api/telemetry/config")
        assert response.status_code == 500
        assert "Failed to retrieve" in response.json()["detail"]

    def test_update_exception_handling(self, client, mock_mavlink_service):
        """Test exception handling in update endpoint."""
        mock_mavlink_service.update_telemetry_config.side_effect = Exception("Test error")

        response = client.put("/api/telemetry/config", json={"rssi_rate_hz": 2.0})
        assert response.status_code == 500
        assert "Failed to update" in response.json()["detail"]

    def test_rssi_send_failure(self, client, mock_mavlink_service):
        """Test handling of RSSI send failure."""
        mock_mavlink_service.send_named_value_float.return_value = False

        response = client.post("/api/telemetry/test/rssi", json={"rssi": -80.0})
        assert response.status_code == 500
        assert "Failed to send test RSSI" in response.json()["detail"]
