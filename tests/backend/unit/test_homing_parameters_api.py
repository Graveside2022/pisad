"""
Unit tests for homing parameters API endpoints.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.routes.system import router
from src.backend.core.config import Config, HomingConfig


@pytest.fixture
def app():
    """Create FastAPI app with system router."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock(spec=Config)
    config.homing = HomingConfig(
        HOMING_FORWARD_VELOCITY_MAX=5.0,
        HOMING_YAW_RATE_MAX=0.5,
        HOMING_APPROACH_VELOCITY=1.0,
        HOMING_SIGNAL_LOSS_TIMEOUT=5.0,
        HOMING_ALGORITHM_MODE="GRADIENT",
        HOMING_GRADIENT_WINDOW_SIZE=10,
        HOMING_GRADIENT_MIN_SNR=10.0,
        HOMING_SAMPLING_TURN_RADIUS=10.0,
        HOMING_SAMPLING_DURATION=5.0,
        HOMING_APPROACH_THRESHOLD=-50.0,
        HOMING_PLATEAU_VARIANCE=2.0,
        HOMING_VELOCITY_SCALE_FACTOR=0.1,
    )
    return config


class TestHomingParametersAPI:
    """Test homing parameters API endpoints."""

    def test_get_homing_parameters(self, client, mock_config):
        """Test getting current homing parameters."""
        with patch("src.backend.api.routes.system.get_config", return_value=mock_config):
            response = client.get("/api/homing/parameters")

            assert response.status_code == 200
            data = response.json()

            assert "parameters" in data
            assert "timestamp" in data

            params = data["parameters"]
            assert params["forward_velocity_max"] == 5.0
            assert params["yaw_rate_max"] == 0.5
            assert params["approach_velocity"] == 1.0
            assert params["signal_loss_timeout"] == 5.0
            assert params["algorithm_mode"] == "GRADIENT"
            assert params["gradient_window_size"] == 10
            assert params["gradient_min_snr"] == 10.0
            assert params["sampling_turn_radius"] == 10.0
            assert params["sampling_duration"] == 5.0
            assert params["approach_threshold"] == -50.0
            assert params["plateau_variance"] == 2.0
            assert params["velocity_scale_factor"] == 0.1

    def test_update_homing_parameters_partial(self, client, mock_config):
        """Test partial update of homing parameters."""
        with patch("src.backend.api.routes.system.get_config", return_value=mock_config):
            update_data = {
                "forward_velocity_max": 7.0,
                "approach_threshold": -45.0,
            }

            response = client.patch("/api/homing/parameters", json=update_data)

            assert response.status_code == 200
            data = response.json()

            # Check updated values
            params = data["parameters"]
            assert params["forward_velocity_max"] == 7.0
            assert params["approach_threshold"] == -45.0

            # Check unchanged values
            assert params["yaw_rate_max"] == 0.5
            assert params["gradient_window_size"] == 10

            # Verify config was updated
            assert mock_config.homing.HOMING_FORWARD_VELOCITY_MAX == 7.0
            assert mock_config.homing.HOMING_APPROACH_THRESHOLD == -45.0

    def test_update_homing_parameters_validation(self, client, mock_config):
        """Test parameter validation in update."""
        with patch("src.backend.api.routes.system.get_config", return_value=mock_config):
            # Test invalid forward_velocity_max (too high)
            response = client.patch(
                "/api/homing/parameters", json={"forward_velocity_max": 15.0}  # Max is 10.0
            )
            assert response.status_code == 422

            # Test invalid approach_threshold (too high)
            response = client.patch(
                "/api/homing/parameters", json={"approach_threshold": -10.0}  # Max is -20.0
            )
            assert response.status_code == 422

            # Test invalid gradient_window_size (too small)
            response = client.patch(
                "/api/homing/parameters", json={"gradient_window_size": 2}  # Min is 3
            )
            assert response.status_code == 422

    def test_update_homing_parameters_all(self, client, mock_config):
        """Test updating all homing parameters."""
        with patch("src.backend.api.routes.system.get_config", return_value=mock_config):
            update_data = {
                "forward_velocity_max": 8.0,
                "yaw_rate_max": 0.8,
                "approach_velocity": 2.0,
                "signal_loss_timeout": 10.0,
                "gradient_window_size": 20,
                "gradient_min_snr": 15.0,
                "sampling_turn_radius": 15.0,
                "sampling_duration": 8.0,
                "approach_threshold": -55.0,
                "plateau_variance": 3.0,
                "velocity_scale_factor": 0.2,
            }

            response = client.patch("/api/homing/parameters", json=update_data)

            assert response.status_code == 200
            data = response.json()

            params = data["parameters"]
            for key, value in update_data.items():
                # Convert snake_case to match response format
                assert params[key] == value

    def test_get_homing_parameters_error(self, client):
        """Test error handling in get parameters."""
        with patch(
            "src.backend.api.routes.system.get_config", side_effect=Exception("Config error")
        ):
            response = client.get("/api/homing/parameters")
            assert response.status_code == 500
            assert "Config error" in response.json()["detail"]

    def test_update_homing_parameters_error(self, client):
        """Test error handling in update parameters."""
        with patch(
            "src.backend.api.routes.system.get_config", side_effect=Exception("Update error")
        ):
            response = client.patch("/api/homing/parameters", json={"forward_velocity_max": 5.0})
            assert response.status_code == 500
            assert "Update error" in response.json()["detail"]
