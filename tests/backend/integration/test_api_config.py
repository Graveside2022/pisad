"""Integration tests for configuration API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def app():
    """Create test app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_config_service():
    """Mock configuration service."""
    mock = MagicMock()
    mock.get_config = MagicMock(
        return_value={
            "sdr": {"frequency": 406025000, "sample_rate": 2048000, "gain": 40},
            "signal_processing": {"fft_size": 1024, "ewma_alpha": 0.3, "snr_threshold": 12.0},
            "homing": {"algorithm": "GRADIENT", "velocity_scale": 1.0, "approach_distance": 10.0},
        }
    )
    mock.update_config = MagicMock(return_value=True)
    mock.get_config_schema = MagicMock(
        return_value={
            "type": "object",
            "properties": {
                "sdr": {"type": "object"},
                "signal_processing": {"type": "object"},
                "homing": {"type": "object"},
            },
        }
    )
    mock.validate_config = MagicMock(return_value=(True, None))
    mock.reset_to_defaults = MagicMock(return_value=True)
    return mock


class TestConfigurationEndpoints:
    """Test configuration management endpoints."""

    def test_get_configuration(self, client, mock_config_service):
        """Test getting current configuration."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.get("/api/config")

            assert response.status_code == 200
            data = response.json()

            assert "sdr" in data
            assert data["sdr"]["frequency"] == 406025000
            assert "signal_processing" in data
            assert "homing" in data

    def test_update_configuration(self, client, mock_config_service):
        """Test updating configuration."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.put("/api/config", json={"sdr": {"gain": 45}})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            mock_config_service.update_config.assert_called_once()

    def test_update_configuration_invalid(self, client, mock_config_service):
        """Test updating with invalid configuration."""
        mock_config_service.validate_config = MagicMock(return_value=(False, "Invalid value"))

        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.put("/api/config", json={"sdr": {"gain": -100}})  # Invalid gain value

            assert response.status_code == 400
            data = response.json()
            assert "detail" in data

    def test_get_config_schema(self, client, mock_config_service):
        """Test getting configuration schema."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.get("/api/config/schema")

            assert response.status_code == 200
            data = response.json()
            assert "type" in data
            assert data["type"] == "object"
            assert "properties" in data

    def test_validate_configuration(self, client, mock_config_service):
        """Test configuration validation."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.post("/api/config/validate", json={"sdr": {"frequency": 406025000}})

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["errors"] is None

    def test_reset_configuration(self, client, mock_config_service):
        """Test resetting configuration to defaults."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.post("/api/config/reset")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            mock_config_service.reset_to_defaults.assert_called_once()


class TestConfigSectionEndpoints:
    """Test configuration section-specific endpoints."""

    def test_get_sdr_config(self, client, mock_config_service):
        """Test getting SDR configuration."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.get("/api/config/sdr")

            assert response.status_code == 200
            data = response.json()
            assert "frequency" in data
            assert data["frequency"] == 406025000
            assert "sample_rate" in data
            assert "gain" in data

    def test_update_sdr_config(self, client, mock_config_service):
        """Test updating SDR configuration."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.put("/api/config/sdr", json={"gain": 50})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_get_homing_config(self, client, mock_config_service):
        """Test getting homing configuration."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.get("/api/config/homing")

            assert response.status_code == 200
            data = response.json()
            assert "algorithm" in data
            assert data["algorithm"] == "GRADIENT"
            assert "velocity_scale" in data

    def test_update_homing_config(self, client, mock_config_service):
        """Test updating homing configuration."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.put("/api/config/homing", json={"velocity_scale": 1.5})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestConfigExportImport:
    """Test configuration export/import functionality."""

    def test_export_configuration(self, client, mock_config_service):
        """Test exporting configuration."""
        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.get("/api/config/export")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

            data = response.json()
            assert "sdr" in data
            assert "signal_processing" in data
            assert "homing" in data

    def test_import_configuration(self, client, mock_config_service):
        """Test importing configuration."""
        config_data = {
            "sdr": {"frequency": 406025000, "sample_rate": 2048000, "gain": 45},
            "signal_processing": {"fft_size": 2048, "ewma_alpha": 0.2, "snr_threshold": 15.0},
        }

        with patch("src.backend.api.routes.config.config_service", mock_config_service):
            response = client.post("/api/config/import", json=config_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            mock_config_service.update_config.assert_called_once_with(config_data)
