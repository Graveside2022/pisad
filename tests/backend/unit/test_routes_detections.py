"""Test suite for detection API routes."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for detection routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock detection service dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        mock_signal_processor = Mock()
        mock_service_manager.signal_processor = mock_signal_processor

        yield {"signal_processor": mock_signal_processor, "service_manager": mock_service_manager}


class TestDetectionEndpoints:
    """Test detection endpoints per PRD-FR1."""

    def test_get_current_detections(self, client, mock_services):
        """Test current detections retrieval per PRD-FR1."""
        mock_detections = {
            "active_detections": [
                {
                    "detection_id": "det_001",
                    "frequency": 3200000000,  # 3.2 GHz per PRD-FR1
                    "rssi": -65.2,
                    "snr": 15.8,  # >12 dB threshold per PRD-FR1
                    "confidence": 0.95,
                    "timestamp": "2025-08-17T15:00:00Z",
                    "location": {"lat": 40.7128, "lon": -74.0060},
                }
            ],
            "detection_count": 1,
            "detection_rate": 95.8,
        }

        mock_services["signal_processor"].get_current_detections.return_value = mock_detections
        response = client.get("/api/detections/current")
        assert response.status_code == 200

    def test_get_detection_history(self, client, mock_services):
        """Test detection history retrieval."""
        mock_history = {
            "detections": [
                {
                    "detection_id": "det_001",
                    "timestamp": "2025-08-17T15:00:00Z",
                    "frequency": 3200000000,
                    "rssi": -65.2,
                    "confidence": 0.95,
                },
                {
                    "detection_id": "det_002",
                    "timestamp": "2025-08-17T15:01:00Z",
                    "frequency": 3200000000,
                    "rssi": -63.8,
                    "confidence": 0.97,
                },
            ],
            "total_detections": 2,
            "time_range": {"start": "2025-08-17T15:00:00Z", "end": "2025-08-17T15:01:00Z"},
        }

        mock_services["signal_processor"].get_detection_history.return_value = mock_history
        response = client.get("/api/detections/history?limit=100")
        assert response.status_code == 200

    def test_detection_statistics(self, client, mock_services):
        """Test detection statistics endpoint."""
        mock_stats = {
            "total_detections": 25,
            "detection_rate": 95.8,
            "false_positive_rate": 2.1,  # <5% per PRD-NFR7
            "average_snr": 16.2,
            "strongest_signal": -58.5,
            "weakest_signal": -72.1,
            "frequency_distribution": {"3200000000": 20, "2437000000": 5},
        }

        mock_services["signal_processor"].get_detection_stats.return_value = mock_stats
        response = client.get("/api/detections/statistics")
        assert response.status_code == 200

    def test_configure_detection_parameters(self, client, mock_services):
        """Test detection parameter configuration."""
        config_request = {
            "rssi_threshold": -70.0,
            "snr_threshold": 12.0,  # Per PRD-FR1
            "confidence_threshold": 0.90,
            "frequency_range": {"min": 850000000, "max": 6500000000},  # Per PRD-FR1
        }

        mock_services["signal_processor"].configure_detection.return_value = {
            "status": "configured"
        }
        response = client.post("/api/detections/configure", json=config_request)
        assert response.status_code == 200

    def test_clear_detection_history(self, client, mock_services):
        """Test detection history clearing."""
        clear_request = {"before_date": "2025-08-17T00:00:00Z"}

        mock_services["signal_processor"].clear_detection_history.return_value = {
            "status": "cleared",
            "records_removed": 150,
        }

        response = client.delete("/api/detections/history", json=clear_request)
        assert response.status_code == 200
