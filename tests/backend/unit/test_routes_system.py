"""
Test suite for system API routes.

Validates system control, configuration, hardware management,
and operational monitoring per PRD requirements.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for system routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock system service dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        # Mock system services
        mock_hardware_detector = Mock()
        mock_config_service = Mock()
        mock_safety_manager = Mock()

        mock_service_manager.hardware_detector = mock_hardware_detector
        mock_service_manager.config_service = mock_config_service
        mock_service_manager.safety_manager = mock_safety_manager

        yield {
            "hardware": mock_hardware_detector,
            "config": mock_config_service,
            "safety": mock_safety_manager,
            "service_manager": mock_service_manager,
        }


class TestSystemStatus:
    """Test system status and monitoring endpoints."""

    def test_get_system_status_healthy(self, client, mock_services):
        """Test system status retrieval when all systems healthy."""
        # Mock healthy system status
        mock_status = {
            "overall_status": "healthy",
            "services": {
                "sdr_service": "running",
                "mavlink_service": "running",
                "state_machine": "running",
                "signal_processor": "running",
            },
            "hardware": {"hackrf": "connected", "flight_controller": "connected"},
            "performance": {"cpu_usage": 45.2, "memory_usage": 67.8, "disk_usage": 23.1},
            "uptime": 3600.0,
        }

        mock_services["service_manager"].get_system_status.return_value = mock_status

        response = client.get("/api/system/status")

        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "healthy"
        assert "services" in data
        assert "hardware" in data
        assert "performance" in data

    def test_system_restart_operation(self, client, mock_services):
        """Test system restart functionality."""
        restart_request = {"component": "all", "graceful": True}

        mock_services["service_manager"].restart_system.return_value = {
            "status": "restart_initiated",
            "estimated_downtime": 30.0,
        }

        response = client.post("/api/system/restart", json=restart_request)

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "estimated_downtime" in data

    def test_system_shutdown_operation(self, client, mock_services):
        """Test system shutdown functionality."""
        shutdown_request = {"graceful": True, "delay_seconds": 10}

        mock_services["service_manager"].shutdown_system.return_value = {
            "status": "shutdown_initiated"
        }

        response = client.post("/api/system/shutdown", json=shutdown_request)

        assert response.status_code == 200


class TestHardwareManagement:
    """Test hardware detection and management per PRD requirements."""

    def test_hardware_detection_success(self, client, mock_services):
        """Test hardware detection per PRD hardware requirements."""
        # Mock hardware detection results per PRD specifications
        mock_hardware = {
            "sdr": {
                "status": "detected",
                "device": "HackRF One",
                "frequency_range": "850MHz - 6.5GHz",  # Per PRD-FR1
                "serial": "66a062dc2227359f",
            },
            "flight_controller": {
                "status": "detected",
                "device": "Cube Orange+",  # Per PRD specs
                "firmware": "ArduPilot 4.3.0",
                "connection": "/dev/ttyACM0",
            },
            "companion_computer": {
                "status": "running",
                "device": "Raspberry Pi 5",
                "cpu_temp": 45.2,
                "available_memory": 2048,
            },
        }

        mock_services["hardware"].detect_hardware.return_value = mock_hardware

        response = client.get("/api/system/hardware")

        assert response.status_code == 200
        data = response.json()
        assert "sdr" in data
        assert "flight_controller" in data
        assert data["sdr"]["frequency_range"] == "850MHz - 6.5GHz"

    def test_hardware_calibration_sdr(self, client, mock_services):
        """Test SDR calibration functionality per PRD-FR1."""
        calibration_request = {
            "device": "sdr",
            "calibration_type": "frequency",
            "parameters": {
                "center_frequency": 3200000000,  # 3.2 GHz default
                "sample_rate": 20000000,
                "gain": 16,
            },
        }

        mock_services["hardware"].calibrate_device.return_value = {
            "status": "calibration_complete",
            "results": {"frequency_accuracy": 99.97, "signal_quality": 95.2},
        }

        response = client.post("/api/system/hardware/calibrate", json=calibration_request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "calibration_complete"

    def test_hardware_reset_device(self, client, mock_services):
        """Test hardware device reset functionality."""
        reset_request = {"device": "sdr", "soft_reset": True}

        mock_services["hardware"].reset_device.return_value = {
            "status": "reset_complete",
            "device_ready": True,
        }

        response = client.post("/api/system/hardware/reset", json=reset_request)

        assert response.status_code == 200


class TestSafetyControls:
    """Test safety control systems per PRD safety requirements."""

    def test_emergency_stop_activation(self, client, mock_services):
        """Test emergency stop functionality per PRD-FR16."""
        emergency_request = {"reason": "operator_initiated", "immediate": True}

        mock_services["safety"].emergency_stop.return_value = {
            "status": "emergency_stop_activated",
            "response_time_ms": 85,  # Should be <500ms per PRD-FR16
            "systems_stopped": ["homing", "velocity_commands", "autonomous_flight"],
        }

        response = client.post("/api/system/emergency/stop", json=emergency_request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "emergency_stop_activated"
        assert data["response_time_ms"] < 500  # PRD-FR16 requirement

    def test_safety_mode_activation(self, client, mock_services):
        """Test safety mode activation per PRD-FR10."""
        safety_request = {
            "mode": "rtl",  # Return to Launch per PRD-FR10
            "trigger": "communication_loss",
        }

        mock_services["safety"].activate_safety_mode.return_value = {
            "status": "safety_mode_active",
            "mode": "rtl",
            "estimated_return_time": 180.0,
        }

        response = client.post("/api/system/safety/activate", json=safety_request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "safety_mode_active"

    def test_get_safety_status(self, client, mock_services):
        """Test safety system status monitoring."""
        mock_safety_status = {
            "emergency_stop_armed": True,
            "geofence_active": True,
            "battery_level": 85.2,
            "communication_quality": 98.5,
            "flight_mode": "GUIDED",
            "safety_violations": [],
        }

        mock_services["safety"].get_safety_status.return_value = mock_safety_status

        response = client.get("/api/system/safety/status")

        assert response.status_code == 200
        data = response.json()
        assert "emergency_stop_armed" in data
        assert "geofence_active" in data


class TestSystemConfiguration:
    """Test system configuration management."""

    def test_get_system_configuration(self, client, mock_services):
        """Test retrieval of system configuration."""
        mock_config = {
            "flight_parameters": {
                "max_velocity": 10.0,  # PRD-FR2 maximum
                "search_altitude": 50.0,
                "geofence_radius": 1000.0,
            },
            "signal_processing": {
                "rssi_threshold": 12.0,  # PRD-FR1 >12 dB SNR
                "processing_latency_target": 100,  # PRD-NFR2 <100ms
                "noise_floor_method": "10th_percentile",
            },
            "safety_settings": {
                "emergency_stop_timeout": 500,  # PRD-FR16 <500ms
                "battery_rtl_threshold": 20.0,
                "communication_timeout": 10.0,
            },
        }

        mock_services["config"].get_system_config.return_value = mock_config

        response = client.get("/api/system/config")

        assert response.status_code == 200
        data = response.json()
        assert "flight_parameters" in data
        assert "signal_processing" in data
        assert data["flight_parameters"]["max_velocity"] == 10.0

    def test_update_system_configuration(self, client, mock_services):
        """Test system configuration updates."""
        config_update = {"flight_parameters": {"search_altitude": 75.0, "search_velocity": 7.5}}

        mock_services["config"].update_config.return_value = {
            "status": "config_updated",
            "restart_required": False,
        }

        response = client.patch("/api/system/config", json=config_update)

        assert response.status_code == 200

    def test_configuration_validation(self, client, mock_services):
        """Test configuration parameter validation."""
        invalid_config = {
            "flight_parameters": {"max_velocity": 25.0}  # Exceeds PRD-FR2 maximum of 10 m/s
        }

        response = client.patch("/api/system/config", json=invalid_config)

        # Should fail validation
        assert response.status_code == 422


class TestSystemLogs:
    """Test system logging and diagnostics."""

    def test_get_system_logs(self, client, mock_services):
        """Test system log retrieval."""
        mock_logs = {
            "logs": [
                {
                    "timestamp": "2025-08-17T15:00:00Z",
                    "level": "INFO",
                    "component": "sdr_service",
                    "message": "HackRF initialized successfully",
                },
                {
                    "timestamp": "2025-08-17T15:00:01Z",
                    "level": "INFO",
                    "component": "mavlink_service",
                    "message": "Flight controller connected",
                },
            ],
            "total_entries": 2,
            "log_level": "INFO",
        }

        mock_services["service_manager"].get_system_logs.return_value = mock_logs

        response = client.get("/api/system/logs?limit=100&level=INFO")

        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert len(data["logs"]) == 2

    def test_clear_system_logs(self, client, mock_services):
        """Test system log clearing functionality."""
        clear_request = {"before_date": "2025-08-17T00:00:00Z", "log_level": "DEBUG"}

        mock_services["service_manager"].clear_logs.return_value = {
            "status": "logs_cleared",
            "entries_removed": 1250,
        }

        response = client.delete("/api/system/logs", json=clear_request)

        assert response.status_code == 200
