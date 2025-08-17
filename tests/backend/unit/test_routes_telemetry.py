"""Test suite for telemetry API routes."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for telemetry routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock telemetry service dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        mock_mavlink_service = Mock()
        mock_telemetry_recorder = Mock()

        mock_service_manager.mavlink_service = mock_mavlink_service
        mock_service_manager.telemetry_recorder = mock_telemetry_recorder

        yield {
            "mavlink": mock_mavlink_service,
            "telemetry": mock_telemetry_recorder,
            "service_manager": mock_service_manager,
        }


class TestTelemetryStreaming:
    """Test telemetry streaming endpoints per PRD-FR9."""

    def test_get_current_telemetry(self, client, mock_services):
        """Test current telemetry retrieval per PRD-FR9."""
        mock_telemetry = {
            "position": {"lat": 40.7128, "lon": -74.0060, "alt": 50.0},
            "attitude": {"roll": 0.1, "pitch": -0.2, "yaw": 1.57},
            "velocity": {"vx": 5.0, "vy": 0.0, "vz": 0.0},
            "battery": {"voltage": 22.4, "current": 8.5, "remaining": 75},
            "flight_mode": "GUIDED",
            "armed": True,
            "gps_status": "3D_FIX",
            "timestamp": "2025-08-17T15:00:00Z",
        }

        mock_services["mavlink"].get_current_telemetry.return_value = mock_telemetry
        response = client.get("/api/telemetry/current")
        assert response.status_code == 200

    def test_get_telemetry_stream(self, client, mock_services):
        """Test telemetry stream endpoint."""
        mock_stream_data = {
            "stream_id": "telemetry_001",
            "frequency": 10,  # 10 Hz
            "data_types": ["position", "attitude", "velocity"],
            "status": "active",
        }

        mock_services["telemetry"].get_stream_info.return_value = mock_stream_data
        response = client.get("/api/telemetry/stream")
        assert response.status_code == 200

    def test_configure_telemetry_stream(self, client, mock_services):
        """Test telemetry stream configuration."""
        stream_config = {
            "frequency": 5,  # 5 Hz
            "data_types": ["position", "battery"],
            "format": "json",
        }

        mock_services["telemetry"].configure_stream.return_value = {"status": "configured"}
        response = client.post("/api/telemetry/stream/configure", json=stream_config)
        assert response.status_code == 200

    def test_get_telemetry_history(self, client, mock_services):
        """Test telemetry history retrieval."""
        mock_history = {
            "telemetry_data": [
                {
                    "timestamp": "2025-08-17T15:00:00Z",
                    "position": {"lat": 40.7128, "lon": -74.0060},
                },
                {
                    "timestamp": "2025-08-17T15:00:01Z",
                    "position": {"lat": 40.7129, "lon": -74.0059},
                },
            ],
            "total_records": 2,
            "time_range": {"start": "2025-08-17T15:00:00Z", "end": "2025-08-17T15:00:01Z"},
        }

        mock_services["telemetry"].get_history.return_value = mock_history
        response = client.get(
            "/api/telemetry/history?start_time=2025-08-17T15:00:00Z&end_time=2025-08-17T15:00:01Z"
        )
        assert response.status_code == 200

    def test_rssi_telemetry_streaming(self, client, mock_services):
        """Test RSSI telemetry streaming per PRD-FR9."""
        mock_rssi_data = {
            "current_rssi": -65.2,
            "rssi_history": [-67.1, -66.8, -65.9, -65.2],
            "snr": 15.8,
            "frequency": 3200000000,
            "signal_detected": True,
            "timestamp": "2025-08-17T15:00:00Z",
        }

        mock_services["telemetry"].get_rssi_data.return_value = mock_rssi_data
        response = client.get("/api/telemetry/rssi")
        assert response.status_code == 200
