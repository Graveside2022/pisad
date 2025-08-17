"""Test suite for configuration API routes."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for config routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock config service dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        mock_config_service = Mock()
        mock_service_manager.config_service = mock_config_service

        yield {"config": mock_config_service, "service_manager": mock_service_manager}


class TestConfigurationManagement:
    """Test configuration management endpoints."""

    def test_get_current_config(self, client, mock_services):
        """Test retrieval of current configuration."""
        mock_config = {
            "flight_parameters": {"max_velocity": 10.0, "search_altitude": 50.0},
            "signal_processing": {"rssi_threshold": 12.0, "processing_latency_target": 100},
            "safety_settings": {"emergency_stop_timeout": 500, "battery_rtl_threshold": 20.0},
        }

        mock_services["config"].get_config.return_value = mock_config
        response = client.get("/api/config")
        assert response.status_code == 200

    def test_update_config_success(self, client, mock_services):
        """Test successful configuration update."""
        config_update = {"flight_parameters": {"search_altitude": 75.0}}

        mock_services["config"].update_config.return_value = {"status": "updated"}
        response = client.patch("/api/config", json=config_update)
        assert response.status_code == 200

    def test_get_config_profiles(self, client, mock_services):
        """Test configuration profile listing."""
        mock_profiles = {"profiles": ["default", "custom", "test"]}

        mock_services["config"].get_profiles.return_value = mock_profiles
        response = client.get("/api/config/profiles")
        assert response.status_code == 200

    def test_load_config_profile(self, client, mock_services):
        """Test loading configuration profile."""
        profile_request = {"profile_name": "custom"}

        mock_services["config"].load_profile.return_value = {"status": "loaded"}
        response = client.post("/api/config/profiles/load", json=profile_request)
        assert response.status_code == 200

    def test_save_config_profile(self, client, mock_services):
        """Test saving configuration profile."""
        save_request = {"profile_name": "test_profile", "config": {"test": "value"}}

        mock_services["config"].save_profile.return_value = {"status": "saved"}
        response = client.post("/api/config/profiles/save", json=save_request)
        assert response.status_code == 200
