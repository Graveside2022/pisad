"""
Test network configuration API endpoints.

SUBTASK-5.6.2.3 [8e6f] - Tests for runtime configuration update API endpoint
with <500ms response time validation.
"""

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.backend.core.app import app
from src.backend.core.config import Config, NetworkConfig

# Create test client
test_client = TestClient(app)


@pytest.fixture
def mock_network_config():
    """Mock network configuration for testing."""
    return NetworkConfig(
        NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.01,
        NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD=0.05,
        NETWORK_PACKET_LOSS_HIGH_THRESHOLD=0.10,
        NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.20,
        NETWORK_CONGESTION_DETECTOR_ENABLED=True,
        NETWORK_BASELINE_LATENCY_MS=0.0,
        NETWORK_LATENCY_THRESHOLD_MS=100.0,
        NETWORK_RUNTIME_ADJUSTMENT_ENABLED=True,
        NETWORK_OPERATOR_OVERRIDE_ENABLED=True,
        NETWORK_MONITORING_INTERVAL_MS=1000,
        NETWORK_ADAPTIVE_RATE_ENABLED=True,
    )


@pytest.fixture
def mock_config(mock_network_config):
    """Mock full configuration for testing."""
    config = Config()
    config.network = mock_network_config
    return config


class TestNetworkConfigAPI:
    """Test network configuration API endpoints."""

    @patch("src.backend.api.routes.config.get_config")
    def test_get_network_config_success(self, mock_get_config, mock_config):
        """Test successful retrieval of network configuration."""
        mock_get_config.return_value = mock_config

        response = test_client.get("/api/config/network")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all expected fields are present
        expected_fields = [
            "low_threshold",
            "medium_threshold",
            "high_threshold",
            "critical_threshold",
            "congestion_detector_enabled",
            "baseline_latency_ms",
            "latency_threshold_ms",
            "runtime_adjustment_enabled",
            "operator_override_enabled",
            "monitoring_interval_ms",
            "adaptive_rate_enabled",
            "update_timestamp",
        ]

        for field in expected_fields:
            assert field in data

        # Verify threshold values
        assert data["low_threshold"] == 0.01
        assert data["medium_threshold"] == 0.05
        assert data["high_threshold"] == 0.10
        assert data["critical_threshold"] == 0.20
        assert data["congestion_detector_enabled"] is True
        assert data["latency_threshold_ms"] == 100.0

    @patch("src.backend.api.routes.config.get_config")
    def test_get_network_config_performance(self, mock_get_config, mock_config):
        """Test that get network config meets performance requirements."""
        mock_get_config.return_value = mock_config

        start_time = time.perf_counter()
        response = test_client.get("/api/config/network")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert response.status_code == status.HTTP_200_OK
        # Should be much faster than 500ms for GET operation
        assert elapsed_ms < 100  # 100ms threshold for GET

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_update_network_config_success(self, mock_get_config, mock_broadcast, mock_config):
        """Test successful update of network configuration."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        update_data = {
            "low_threshold": 0.02,
            "medium_threshold": 0.06,
            "high_threshold": 0.12,
            "critical_threshold": 0.25,
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 150.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        response = test_client.put("/api/config/network", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify updated values
        assert data["low_threshold"] == 0.02
        assert data["medium_threshold"] == 0.06
        assert data["high_threshold"] == 0.12
        assert data["critical_threshold"] == 0.25
        assert data["latency_threshold_ms"] == 150.0

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_update_network_config_performance(self, mock_get_config, mock_broadcast, mock_config):
        """Test that update network config meets <500ms requirement."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        update_data = {
            "low_threshold": 0.015,
            "medium_threshold": 0.055,
            "high_threshold": 0.105,
            "critical_threshold": 0.205,
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 120.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        start_time = time.perf_counter()
        response = test_client.put("/api/config/network", json=update_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert response.status_code == status.HTTP_200_OK
        # Verify <500ms response time requirement
        assert elapsed_ms < 500, f"Update took {elapsed_ms:.2f}ms, exceeding 500ms requirement"

    @patch("src.backend.api.routes.config.get_config")
    def test_update_network_config_invalid_threshold_ordering(self, mock_get_config, mock_config):
        """Test validation of threshold ordering."""
        mock_get_config.return_value = mock_config

        # Invalid ordering: medium < low
        update_data = {
            "low_threshold": 0.10,  # Invalid: higher than medium
            "medium_threshold": 0.05,
            "high_threshold": 0.15,
            "critical_threshold": 0.25,
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 100.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        response = test_client.put("/api/config/network", json=update_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "ascending order" in response.json()["detail"]

    @patch("src.backend.api.routes.config.get_config")
    def test_update_network_config_invalid_threshold_bounds(self, mock_get_config, mock_config):
        """Test validation of threshold bounds (0.001-0.5)."""
        mock_get_config.return_value = mock_config

        # Invalid bounds: threshold too high
        update_data = {
            "low_threshold": 0.01,
            "medium_threshold": 0.05,
            "high_threshold": 0.10,
            "critical_threshold": 0.6,  # Invalid: > 0.5
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 100.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        response = test_client.put("/api/config/network", json=update_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.backend.api.routes.config.get_config")
    def test_update_network_config_runtime_disabled(self, mock_get_config, mock_config):
        """Test update blocked when runtime adjustment is disabled."""
        # Disable runtime adjustment
        mock_config.network.NETWORK_RUNTIME_ADJUSTMENT_ENABLED = False
        mock_get_config.return_value = mock_config

        update_data = {
            "low_threshold": 0.02,
            "medium_threshold": 0.06,
            "high_threshold": 0.12,
            "critical_threshold": 0.25,
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 150.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        response = test_client.put("/api/config/network", json=update_data)

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "Runtime network configuration updates are disabled" in response.json()["detail"]

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_update_individual_threshold_success(
        self, mock_get_config, mock_broadcast, mock_config
    ):
        """Test successful update of individual threshold."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        response = test_client.patch("/api/config/network/thresholds/medium?threshold=0.08")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "success"
        assert data["severity"] == "medium"
        assert data["threshold"] == 0.08
        assert "update_timestamp" in data
        assert "response_time_ms" in data

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_update_individual_threshold_performance(
        self, mock_get_config, mock_broadcast, mock_config
    ):
        """Test individual threshold update meets <500ms requirement."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        start_time = time.perf_counter()
        response = test_client.patch("/api/config/network/thresholds/high?threshold=0.15")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert response.status_code == status.HTTP_200_OK
        # Verify <500ms response time requirement
        assert (
            elapsed_ms < 500
        ), f"Threshold update took {elapsed_ms:.2f}ms, exceeding 500ms requirement"

    @patch("src.backend.api.routes.config.get_config")
    def test_update_individual_threshold_invalid_severity(self, mock_get_config, mock_config):
        """Test validation of invalid severity level."""
        mock_get_config.return_value = mock_config

        response = test_client.patch("/api/config/network/thresholds/invalid?threshold=0.08")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid severity level" in response.json()["detail"]

    @patch("src.backend.api.routes.config.get_config")
    def test_update_individual_threshold_invalid_bounds(self, mock_get_config, mock_config):
        """Test validation of threshold bounds for individual update."""
        mock_get_config.return_value = mock_config

        # Test threshold too low
        response = test_client.patch("/api/config/network/thresholds/low?threshold=0.0005")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

        # Test threshold too high
        response = test_client.patch("/api/config/network/thresholds/high?threshold=0.8")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch("src.backend.api.routes.config.get_config")
    def test_update_individual_threshold_ordering_violation(self, mock_get_config, mock_config):
        """Test validation of threshold ordering for individual update."""
        mock_get_config.return_value = mock_config

        # Try to set medium threshold higher than high threshold
        response = test_client.patch("/api/config/network/thresholds/medium?threshold=0.15")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "threshold ordering" in response.json()["detail"]

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_websocket_broadcast_on_update(self, mock_get_config, mock_broadcast, mock_config):
        """Test that configuration updates trigger WebSocket broadcasts."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        update_data = {
            "low_threshold": 0.02,
            "medium_threshold": 0.06,
            "high_threshold": 0.12,
            "critical_threshold": 0.25,
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 150.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        response = test_client.put("/api/config/network", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        # Verify WebSocket broadcast was called
        mock_broadcast.assert_called_once()

        # Verify broadcast message structure
        call_args = mock_broadcast.call_args[0][0]
        assert call_args["type"] == "config"
        assert call_args["action"] == "network_config_updated"
        assert "config" in call_args

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_websocket_broadcast_on_threshold_update(
        self, mock_get_config, mock_broadcast, mock_config
    ):
        """Test that individual threshold updates trigger WebSocket broadcasts."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        response = test_client.patch("/api/config/network/thresholds/critical?threshold=0.30")

        assert response.status_code == status.HTTP_200_OK
        # Verify WebSocket broadcast was called
        mock_broadcast.assert_called_once()

        # Verify broadcast message structure
        call_args = mock_broadcast.call_args[0][0]
        assert call_args["type"] == "config"
        assert call_args["action"] == "network_threshold_updated"
        assert call_args["threshold"]["severity"] == "critical"
        assert call_args["threshold"]["value"] == 0.30


class TestNetworkConfigIntegration:
    """Integration tests for network configuration API."""

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_full_configuration_workflow(self, mock_get_config, mock_broadcast, mock_config):
        """Test complete workflow: get -> update -> verify -> individual update."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        # 1. Get initial configuration
        response = test_client.get("/api/config/network")
        assert response.status_code == status.HTTP_200_OK
        initial_config = response.json()

        # 2. Update configuration
        update_data = {
            "low_threshold": 0.025,
            "medium_threshold": 0.075,
            "high_threshold": 0.125,
            "critical_threshold": 0.275,
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 150.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        response = test_client.put("/api/config/network", json=update_data)
        assert response.status_code == status.HTTP_200_OK
        updated_config = response.json()

        # Verify updates applied
        assert updated_config["low_threshold"] != initial_config["low_threshold"]
        assert updated_config["low_threshold"] == 0.025

        # 3. Update individual threshold
        response = test_client.patch("/api/config/network/thresholds/high?threshold=0.135")
        assert response.status_code == status.HTTP_200_OK

        # Verify individual update
        assert response.json()["threshold"] == 0.135

    @patch("src.backend.api.routes.config.get_config")
    def test_error_handling_and_recovery(self, mock_get_config, mock_config):
        """Test error handling and system recovery."""
        mock_get_config.return_value = mock_config

        # Test invalid data doesn't break system
        invalid_update = {
            "low_threshold": "invalid",  # Wrong type
            "medium_threshold": 0.05,
            "high_threshold": 0.10,
            "critical_threshold": 0.20,
            "congestion_detector_enabled": True,
            "latency_threshold_ms": 100.0,
            "runtime_adjustment_enabled": True,
            "adaptive_rate_enabled": True,
        }

        response = test_client.put("/config/network", json=invalid_update)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Verify system still works after error
        response = test_client.get("/api/config/network")
        assert response.status_code == status.HTTP_200_OK

    def test_api_documentation_and_validation(self):
        """Test that API endpoints are properly documented and validated."""
        # Test OpenAPI schema includes our endpoints
        response = test_client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        openapi_schema = response.json()
        paths = openapi_schema["paths"]

        # Verify network endpoints are documented
        assert "/api/config/network" in paths
        assert "/api/config/network/thresholds/{severity}" in paths

        # Verify HTTP methods
        assert "get" in paths["/api/config/network"]
        assert "put" in paths["/api/config/network"]
        assert "patch" in paths["/api/config/network/thresholds/{severity}"]


@pytest.mark.performance
class TestNetworkConfigPerformance:
    """Performance-focused tests for network configuration API."""

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_concurrent_threshold_updates(self, mock_get_config, mock_broadcast, mock_config):
        """Test performance under concurrent threshold updates."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        import concurrent.futures

        def update_threshold(severity, threshold):
            """Update a threshold and return response time."""
            start_time = time.perf_counter()
            response = test_client.patch(
                f"/api/config/network/thresholds/{severity}?threshold={threshold}"
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return response.status_code, elapsed_ms

        # Test concurrent updates
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(update_threshold, "low", 0.02),
                executor.submit(update_threshold, "medium", 0.06),
                executor.submit(update_threshold, "high", 0.12),
                executor.submit(update_threshold, "critical", 0.25),
            ]

            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All should succeed and meet performance requirements
        for status_code, elapsed_ms in results:
            assert status_code == status.HTTP_200_OK
            assert elapsed_ms < 500  # <500ms requirement

    @patch("src.backend.api.routes.config.broadcast_message")
    @patch("src.backend.api.routes.config.get_config")
    def test_stress_testing_rapid_updates(self, mock_get_config, mock_broadcast, mock_config):
        """Test system stability under rapid configuration updates."""
        mock_get_config.return_value = mock_config
        mock_broadcast.return_value = AsyncMock()

        # Perform rapid successive updates
        response_times = []

        for i in range(10):
            threshold = 0.02 + (i * 0.01)  # 0.02, 0.03, 0.04...

            start_time = time.perf_counter()
            response = test_client.patch(f"/config/network/thresholds/low?threshold={threshold}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            assert response.status_code == status.HTTP_200_OK
            response_times.append(elapsed_ms)

        # All updates should meet performance requirements
        for elapsed_ms in response_times:
            assert elapsed_ms < 500

        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 100  # Average should be much better than 500ms
