"""Test suite for health monitoring API routes."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for health routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock health service dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        yield {"service_manager": mock_service_manager}


class TestHealthMonitoring:
    """Test health monitoring endpoints."""

    def test_health_check_healthy(self, client, mock_services):
        """Test health check when system is healthy."""
        mock_health = {
            "status": "healthy",
            "checks": {
                "database": "healthy",
                "sdr_service": "healthy",
                "mavlink_service": "healthy",
                "signal_processor": "healthy",
            },
            "uptime": 3600.0,
            "timestamp": "2025-08-17T15:00:00Z",
        }

        mock_services["service_manager"].get_health_status.return_value = mock_health
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_check_degraded(self, client, mock_services):
        """Test health check when system is degraded."""
        mock_health = {
            "status": "degraded",
            "checks": {
                "database": "healthy",
                "sdr_service": "healthy",
                "mavlink_service": "unhealthy",  # Flight controller issue
                "signal_processor": "healthy",
            },
            "issues": ["mavlink_service: connection timeout"],
            "uptime": 1800.0,
        }

        mock_services["service_manager"].get_health_status.return_value = mock_health
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_detailed_health_check(self, client, mock_services):
        """Test detailed health check endpoint."""
        mock_detailed_health = {
            "overall_status": "healthy",
            "services": {
                "sdr_service": {"status": "healthy", "latency_ms": 25.5, "cpu_usage": 15.2},
                "mavlink_service": {
                    "status": "healthy",
                    "packet_loss": 0.1,
                    "connection_quality": 98.5,
                },
                "signal_processor": {
                    "status": "healthy",
                    "processing_latency": 45.0,
                    "queue_depth": 3,
                },
            },
            "system_metrics": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "temperature": 42.5,
            },
        }

        mock_services["service_manager"].get_detailed_health.return_value = mock_detailed_health
        response = client.get("/api/health/detailed")
        assert response.status_code == 200

    def test_service_health_individual(self, client, mock_services):
        """Test individual service health check."""
        service_name = "sdr_service"
        mock_service_health = {
            "service": service_name,
            "status": "healthy",
            "details": {
                "connection_status": "connected",
                "device_temperature": 35.2,
                "signal_quality": 95.8,
                "last_error": None,
            },
        }

        mock_services["service_manager"].get_service_health.return_value = mock_service_health
        response = client.get(f"/api/health/service/{service_name}")
        assert response.status_code == 200

    def test_health_metrics_history(self, client, mock_services):
        """Test health metrics history retrieval."""
        mock_history = {
            "metrics": [
                {"timestamp": "2025-08-17T15:00:00Z", "cpu_usage": 45.2, "memory_usage": 67.8},
                {"timestamp": "2025-08-17T15:01:00Z", "cpu_usage": 47.1, "memory_usage": 68.2},
            ],
            "time_range": {"start": "2025-08-17T15:00:00Z", "end": "2025-08-17T15:01:00Z"},
            "sample_count": 2,
        }

        mock_services["service_manager"].get_health_history.return_value = mock_history
        response = client.get("/api/health/metrics/history?duration=3600")
        assert response.status_code == 200
