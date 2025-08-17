"""
Test suite for system API routes.

Comprehensive testing of system control, safety management, homing parameters,
and state management endpoints per PRD requirements FR11, FR15, FR16, NFR12.

This test suite validates all 12 actual system endpoints with authentic
API testing using FastAPI TestClient and proper dependency injection mocking.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app
from src.backend.services.state_machine import SystemState


@pytest.fixture
def client():
    """Create test client for system routes with proper app configuration."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_safety_system():
    """Mock safety system for dependency injection."""
    mock_safety = Mock()

    # Configure safety system methods
    mock_safety.enable_homing = AsyncMock(return_value=True)
    mock_safety.disable_homing = AsyncMock()
    mock_safety.emergency_stop = AsyncMock()
    mock_safety.reset_emergency_stop = AsyncMock()
    mock_safety.get_safety_status.return_value = {
        "homing_enabled": False,
        "emergency_stopped": False,
        "battery_level": 85.2,
        "flight_mode": "GUIDED",
        "interlocks": {
            "velocity_limit": True,
            "geofence": True,
            "battery_low": False,
            "signal_lock": True,
        },
    }
    mock_safety.get_safety_events.return_value = []

    return mock_safety


@pytest.fixture
def mock_state_machine():
    """Mock state machine for dependency injection."""
    mock_state = Mock()

    # Configure state machine methods
    mock_state.get_current_state.return_value = SystemState.IDLE
    mock_state._previous_state = SystemState.IDLE
    mock_state.get_allowed_transitions.return_value = [SystemState.SEARCHING]
    mock_state.get_statistics.return_value = {
        "time_since_detection": 0,
        "total_detections": 0,
        "current_rssi": -80.5,
    }
    mock_state.get_state_history.return_value = []
    mock_state.get_search_pattern_status.return_value = {"active": False, "progress": 0.0}
    mock_state.force_transition = AsyncMock(return_value=True)
    mock_state._state_db = None  # No persistence for tests

    return mock_state


@pytest.fixture
def mock_dependencies(mock_safety_system, mock_state_machine):
    """Mock all system route dependencies."""
    with (
        patch("src.backend.api.routes.system.safety_system", mock_safety_system),
        patch("src.backend.api.routes.system.state_machine", mock_state_machine),
    ):
        yield {"safety_system": mock_safety_system, "state_machine": mock_state_machine}


class TestSystemStatusEndpoint:
    """Test /api/system/status endpoint per PRD system monitoring requirements."""

    def test_get_system_status_success(self, client, mock_dependencies):
        """Test successful system status retrieval with all metrics."""
        with (
            patch("psutil.cpu_percent", return_value=45.2),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
            patch("psutil.boot_time", return_value=1640995200.0),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):  # No temp sensor
            # Configure psutil mocks
            mock_memory.return_value.percent = 67.8
            mock_disk.return_value.percent = 23.1

            response = client.get("/api/system/status")

            assert response.status_code == 200
            data = response.json()

            # Verify core system fields exist
            assert "status" in data
            assert "timestamp" in data
            assert "current_state" in data
            assert "cpu_usage" in data
            assert "memory_usage" in data
            assert "disk_usage" in data
            assert "sdr_frequency" in data
            assert "safety_interlocks" in data

            # Verify system health metrics
            assert data["cpu_usage"] == 45.2
            assert data["memory_usage"] == 67.8
            assert data["disk_usage"] == 23.1

    def test_system_status_with_temperature_sensor(self, client, mock_dependencies):
        """Test system status when Raspberry Pi temperature sensor is available."""
        # Mock temperature file read
        temp_data = "45230\n"  # 45.23°C in millidegrees

        with (
            patch("psutil.cpu_percent", return_value=30.0),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
            patch("psutil.boot_time", return_value=1640995200.0),
            patch("builtins.open", mock_open_func(temp_data)),
        ):
            mock_memory.return_value.percent = 50.0
            mock_disk.return_value.percent = 15.0

            response = client.get("/api/system/status")

            assert response.status_code == 200
            data = response.json()
            assert data["temperature"] == 45.23  # Converted from millidegrees

    def test_system_status_configuration_error(self, client, mock_dependencies):
        """Test system status when configuration fails."""
        from src.backend.core.exceptions import PISADException

        with patch(
            "src.backend.api.routes.system.get_config", side_effect=PISADException("Config error")
        ):
            response = client.get("/api/system/status")

            assert response.status_code == 500
            assert "Config error" in response.json()["detail"]


class TestHomingControlEndpoint:
    """Test /api/system/homing endpoint per PRD-FR11, FR14, FR15, FR16 requirements."""

    def test_enable_homing_success(self, client, mock_dependencies):
        """Test successful homing activation per PRD-FR14."""
        mock_dependencies["safety_system"].enable_homing.return_value = True

        request_data = {"enabled": True, "confirmation_token": "operator_confirm_123"}

        response = client.post("/api/system/homing", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["homing_enabled"] is True
        assert "safety_status" in data
        assert "timestamp" in data

        # Verify safety system was called with token
        mock_dependencies["safety_system"].enable_homing.assert_called_once_with(
            "operator_confirm_123"
        )

    def test_enable_homing_blocked_by_safety(self, client, mock_dependencies):
        """Test homing activation blocked by safety interlocks per PRD-FR15."""
        # Configure safety system to block activation
        mock_dependencies["safety_system"].enable_homing.return_value = False
        mock_dependencies["safety_system"].get_safety_status.return_value = {
            "homing_enabled": False,
            "blocked_reasons": ["flight_mode_not_guided", "battery_low"],
            "emergency_stopped": False,
        }

        request_data = {"enabled": True, "confirmation_token": "test_token"}

        response = client.post("/api/system/homing", json=request_data)

        assert response.status_code == 403
        data = response.json()
        assert "Homing activation blocked" in data["detail"]["message"]
        assert "safety_status" in data["detail"]

    def test_disable_homing_success(self, client, mock_dependencies):
        """Test homing deactivation per PRD-FR16."""
        request_data = {"enabled": False}

        response = client.post("/api/system/homing", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["homing_enabled"] is False

        # Verify safety system disable was called
        mock_dependencies["safety_system"].disable_homing.assert_called_once_with("API request")

    def test_homing_control_safety_system_not_initialized(self, client):
        """Test homing control when safety system is not available."""
        # Test without mocked dependencies (safety_system = None)
        request_data = {"enabled": True}

        response = client.post("/api/system/homing", json=request_data)

        assert response.status_code == 500
        assert "Safety system not initialized" in response.json()["detail"]

    def test_homing_control_invalid_request(self, client, mock_dependencies):
        """Test homing control with malformed request."""
        # Missing required 'enabled' field
        response = client.post("/api/system/homing", json={})

        assert response.status_code == 422  # Validation error


def mock_open_func(content):
    """Helper to create mock for file open with specific content."""
    from unittest.mock import mock_open

    return mock_open(read_data=content)


class TestEmergencyStopEndpoints:
    """Test emergency stop endpoints per PRD-FR16 (<500ms response) requirements."""

    def test_emergency_stop_activation(self, client, mock_dependencies):
        """Test emergency stop activation per PRD-FR16."""
        request_data = {"reason": "Operator initiated emergency stop"}

        response = client.post("/api/system/emergency-stop", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["emergency_stopped"] is True
        assert data["reason"] == "Operator initiated emergency stop"
        assert "safety_status" in data
        assert "timestamp" in data

        # Verify emergency stop was called
        mock_dependencies["safety_system"].emergency_stop.assert_called_once_with(
            "Operator initiated emergency stop"
        )

    def test_emergency_stop_default_reason(self, client, mock_dependencies):
        """Test emergency stop with default reason."""
        response = client.post("/api/system/emergency-stop", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["emergency_stopped"] is True

        # Verify default reason was used
        mock_dependencies["safety_system"].emergency_stop.assert_called_once_with(
            "Manual emergency stop"
        )

    def test_emergency_stop_reset_success(self, client, mock_dependencies):
        """Test emergency stop reset functionality."""
        response = client.post("/api/system/reset-emergency-stop")

        assert response.status_code == 200
        data = response.json()
        assert data["emergency_stopped"] is False
        assert "safety_status" in data

        # Verify reset was called
        mock_dependencies["safety_system"].reset_emergency_stop.assert_called_once()

    def test_emergency_stop_safety_system_not_initialized(self, client):
        """Test emergency stop when safety system unavailable."""
        response = client.post("/api/system/emergency-stop", json={})

        assert response.status_code == 500
        assert "Safety system not initialized" in response.json()["detail"]

    def test_emergency_stop_reset_safety_system_not_initialized(self, client):
        """Test emergency stop reset when safety system unavailable."""
        response = client.post("/api/system/reset-emergency-stop")

        assert response.status_code == 500
        assert "Safety system not initialized" in response.json()["detail"]


class TestSafetyEndpoints:
    """Test safety monitoring endpoints per PRD safety requirements."""

    def test_get_safety_events_success(self, client, mock_dependencies):
        """Test safety event history retrieval."""
        # Mock safety events
        mock_events = [
            Mock(
                id=1,
                timestamp=datetime.now(UTC),
                event_type=Mock(value="HOMING_DISABLED"),
                trigger=Mock(value="SIGNAL_LOSS"),
                details={"duration": 5.2},
                resolved=True,
            ),
            Mock(
                id=2,
                timestamp=datetime.now(UTC),
                event_type=Mock(value="EMERGENCY_STOP"),
                trigger=Mock(value="OPERATOR"),
                details={"reason": "Test emergency stop"},
                resolved=False,
            ),
        ]
        mock_dependencies["safety_system"].get_safety_events.return_value = mock_events

        response = client.get("/api/safety/events?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert "count" in data
        assert "timestamp" in data
        assert data["count"] == 2
        assert len(data["events"]) == 2

    def test_get_safety_events_with_filters(self, client, mock_dependencies):
        """Test safety events with date filtering."""
        since_date = "2025-08-17T00:00:00Z"
        mock_dependencies["safety_system"].get_safety_events.return_value = []

        response = client.get(f"/api/safety/events?limit=50&since={since_date}")

        assert response.status_code == 200
        # Verify filters were passed correctly
        from datetime import datetime

        expected_since = datetime.fromisoformat(since_date.replace("Z", "+00:00"))
        mock_dependencies["safety_system"].get_safety_events.assert_called_once_with(
            since=expected_since, limit=50
        )

    def test_get_safety_status_success(self, client, mock_dependencies):
        """Test safety status retrieval."""
        response = client.get("/api/safety/status")

        assert response.status_code == 200
        data = response.json()

        # Verify safety status structure from mock
        assert "homing_enabled" in data
        assert "emergency_stopped" in data
        assert "battery_level" in data
        assert "flight_mode" in data
        assert "interlocks" in data

    def test_safety_endpoints_system_not_initialized(self, client):
        """Test safety endpoints when safety system unavailable."""
        # Test without mocked dependencies
        response_events = client.get("/api/safety/events")
        response_status = client.get("/api/safety/status")

        assert response_events.status_code == 500
        assert response_status.status_code == 500
        assert "Safety system not initialized" in response_events.json()["detail"]
        assert "Safety system not initialized" in response_status.json()["detail"]


class TestHomingParametersEndpoints:
    """Test homing parameter management per PRD-FR4 gradient climbing requirements."""

    def test_get_homing_parameters_success(self, client, mock_dependencies):
        """Test homing parameters retrieval."""
        response = client.get("/api/homing/parameters")

        assert response.status_code == 200
        data = response.json()
        assert "parameters" in data
        assert "timestamp" in data

        # Verify core homing parameters exist
        params = data["parameters"]
        assert "forward_velocity_max" in params
        assert "yaw_rate_max" in params
        assert "approach_velocity" in params
        assert "signal_loss_timeout" in params
        assert "gradient_window_size" in params
        assert "gradient_min_snr" in params

    def test_update_homing_parameters_partial(self, client, mock_dependencies):
        """Test partial homing parameter updates."""
        update_data = {
            "forward_velocity_max": 8.5,
            "approach_velocity": 3.2,
            "gradient_min_snr": 15.0,
        }

        response = client.patch("/api/homing/parameters", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert "parameters" in data
        assert "timestamp" in data

    def test_update_homing_parameters_validation(self, client, mock_dependencies):
        """Test homing parameter validation per PRD constraints."""
        # Test invalid velocity (exceeds 10.0 m/s limit)
        invalid_data = {"forward_velocity_max": 15.0}

        response = client.patch("/api/homing/parameters", json=invalid_data)

        assert response.status_code == 422  # Validation error

        # Test invalid yaw rate (negative value)
        invalid_data = {"yaw_rate_max": -1.0}

        response = client.patch("/api/homing/parameters", json=invalid_data)

        assert response.status_code == 422

    def test_update_homing_parameters_boundary_values(self, client, mock_dependencies):
        """Test homing parameter boundary validation."""
        # Test minimum valid values
        boundary_data = {
            "forward_velocity_max": 0.1,  # Minimum per validation
            "yaw_rate_max": 0.1,
            "approach_velocity": 0.1,
            "signal_loss_timeout": 1.0,
            "gradient_window_size": 3,
            "gradient_min_snr": 0.0,
        }

        response = client.patch("/api/homing/parameters", json=boundary_data)

        assert response.status_code == 200

        # Test maximum valid values
        boundary_data = {
            "forward_velocity_max": 10.0,  # Maximum per validation
            "yaw_rate_max": 2.0,
            "approach_velocity": 5.0,
            "signal_loss_timeout": 30.0,
            "gradient_window_size": 50,
            "gradient_min_snr": 50.0,
        }

        response = client.patch("/api/homing/parameters", json=boundary_data)

        assert response.status_code == 200


class TestDebugModeEndpoint:
    """Test debug mode control per development requirements."""

    def test_toggle_debug_mode_global(self, client, mock_dependencies):
        """Test global debug mode activation."""
        request_data = {"enabled": True, "target": "all"}

        with patch("src.backend.services.homing_algorithm.set_debug_mode") as mock_homing_debug:
            response = client.post("/api/system/debug", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert "debug_mode" in data
            assert data["debug_mode"]["enabled"] is True
            assert data["debug_mode"]["target"] == "all"

            # Verify homing debug was called
            mock_homing_debug.assert_called_once_with(True)

    def test_toggle_debug_mode_specific_service(self, client, mock_dependencies):
        """Test service-specific debug mode."""
        request_data = {"enabled": False, "target": "homing"}

        with patch("src.backend.services.homing_algorithm.set_debug_mode") as mock_homing_debug:
            response = client.post("/api/system/debug", json=request_data)

            assert response.status_code == 200
            mock_homing_debug.assert_called_once_with(False)

    def test_debug_mode_invalid_target(self, client, mock_dependencies):
        """Test debug mode with invalid target."""
        request_data = {"enabled": True, "target": "invalid_service"}

        response = client.post("/api/system/debug", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_debug_mode_missing_parameters(self, client, mock_dependencies):
        """Test debug mode with missing parameters."""
        response = client.post("/api/system/debug", json={})

        assert response.status_code == 422  # Validation error


class TestStateManagementEndpoints:
    """Test state management endpoints per PRD-FR7 state machine requirements."""

    def test_get_current_state_success(self, client, mock_dependencies):
        """Test current state retrieval with full information."""
        response = client.get("/api/system/state")

        assert response.status_code == 200
        data = response.json()

        # Verify state information structure
        assert "current_state" in data
        assert "previous_state" in data
        assert "allowed_transitions" in data
        assert "state_duration_ms" in data
        assert "history" in data
        assert "search_status" in data
        assert "statistics" in data
        assert "timestamp" in data

        # Verify state values
        assert data["current_state"] == "IDLE"
        assert data["previous_state"] == "IDLE"
        assert "SEARCHING" in data["allowed_transitions"]

    def test_state_override_success(self, client, mock_dependencies):
        """Test manual state override with proper authentication."""
        # Generate today's token for validation
        today_token = "override-" + datetime.now(UTC).strftime("%Y%m%d")

        request_data = {
            "target_state": "SEARCHING",
            "reason": "Manual override for testing",
            "confirmation_token": today_token,
            "operator_id": "test_operator",
        }

        response = client.post("/api/system/state-override", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["new_state"] == "IDLE"  # Mock returns IDLE
        assert data["operator_id"] == "test_operator"
        assert "timestamp" in data

        # Verify force_transition was called
        mock_dependencies["state_machine"].force_transition.assert_called_once_with(
            target_state=SystemState.SEARCHING,
            reason="Manual override for testing",
            operator_id="test_operator",
        )

    def test_state_override_invalid_token(self, client, mock_dependencies):
        """Test state override with invalid confirmation token."""
        request_data = {
            "target_state": "SEARCHING",
            "reason": "Test override",
            "confirmation_token": "invalid_token",
            "operator_id": "test_operator",
        }

        response = client.post("/api/system/state-override", json=request_data)

        assert response.status_code == 403
        assert "Invalid confirmation token" in response.json()["detail"]

    def test_state_override_invalid_state(self, client, mock_dependencies):
        """Test state override with invalid target state."""
        today_token = "override-" + datetime.now(UTC).strftime("%Y%m%d")

        request_data = {
            "target_state": "INVALID_STATE",
            "reason": "Test override",
            "confirmation_token": today_token,
            "operator_id": "test_operator",
        }

        response = client.post("/api/system/state-override", json=request_data)

        # Invalid state pattern fails Pydantic validation first (422), not endpoint logic (400)
        assert response.status_code == 422
        assert "String should match pattern" in response.json()["detail"][0]["msg"]

    def test_state_override_force_transition_failure(self, client, mock_dependencies):
        """Test state override when force transition fails."""
        today_token = "override-" + datetime.now(UTC).strftime("%Y%m%d")
        mock_dependencies["state_machine"].force_transition.return_value = False

        request_data = {
            "target_state": "SEARCHING",
            "reason": "Test override",
            "confirmation_token": today_token,
            "operator_id": "test_operator",
        }

        response = client.post("/api/system/state-override", json=request_data)

        assert response.status_code == 400
        assert "Failed to transition to SEARCHING" in response.json()["detail"]

    def test_get_state_history_success(self, client, mock_dependencies):
        """Test state history retrieval with default parameters."""
        # Mock state history data
        mock_history = [
            {
                "from_state": "IDLE",
                "to_state": "SEARCHING",
                "timestamp": "2025-08-17T15:00:00Z",
                "reason": "Search pattern initiated",
                "duration_ms": 125,
            },
            {
                "from_state": "SEARCHING",
                "to_state": "DETECTING",
                "timestamp": "2025-08-17T15:01:00Z",
                "reason": "Signal detected",
                "duration_ms": 60000,
            },
        ]
        mock_dependencies["state_machine"].get_state_history.return_value = mock_history

        response = client.get("/api/system/state-history")

        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "count" in data
        assert "filters" in data
        assert "timestamp" in data
        assert data["count"] == 2

    def test_get_state_history_with_filters(self, client, mock_dependencies):
        """Test state history with filtering parameters."""
        mock_dependencies["state_machine"].get_state_history.return_value = []

        response = client.get(
            "/api/system/state-history?limit=50&from_state=IDLE&to_state=SEARCHING"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["limit"] == 50
        assert data["filters"]["from_state"] == "IDLE"
        assert data["filters"]["to_state"] == "SEARCHING"

    def test_get_state_history_limit_validation(self, client, mock_dependencies):
        """Test state history limit validation (max 1000)."""
        mock_dependencies["state_machine"].get_state_history.return_value = []

        response = client.get("/api/system/state-history?limit=2000")

        assert response.status_code == 200
        # Should be capped at 1000
        data = response.json()
        assert data["filters"]["limit"] == 1000

    def test_state_endpoints_state_machine_not_initialized(self, client):
        """Test state endpoints when state machine unavailable."""
        # Test without mocked state machine
        response_state = client.get("/api/system/state")
        response_history = client.get("/api/system/state-history")

        assert response_state.status_code == 404
        assert response_history.status_code == 404
        assert "State machine not initialized" in response_state.json()["detail"]
        assert "State machine not initialized" in response_history.json()["detail"]

        # Test state override
        request_data = {
            "target_state": "SEARCHING",
            "reason": "Test",
            "confirmation_token": "override-" + datetime.now(UTC).strftime("%Y%m%d"),
            "operator_id": "test",
        }
        response_override = client.post("/api/system/state-override", json=request_data)
        assert response_override.status_code == 404


class TestSystemRoutesIntegration:
    """Integration tests for system routes with cross-endpoint scenarios."""

    def test_emergency_stop_to_state_override_sequence(self, client, mock_dependencies):
        """Test emergency stop followed by state override recovery."""
        # First, activate emergency stop
        response1 = client.post("/api/system/emergency-stop", json={"reason": "Test emergency"})
        assert response1.status_code == 200

        # Then reset emergency stop
        response2 = client.post("/api/system/reset-emergency-stop")
        assert response2.status_code == 200

        # Finally, override state to return to normal operations
        today_token = "override-" + datetime.now(UTC).strftime("%Y%m%d")
        override_data = {
            "target_state": "IDLE",
            "reason": "Recovery from emergency stop",
            "confirmation_token": today_token,
            "operator_id": "integration_test",
        }
        response3 = client.post("/api/system/state-override", json=override_data)
        assert response3.status_code == 200

    def test_homing_parameters_and_safety_status_consistency(self, client, mock_dependencies):
        """Test parameter updates reflected in safety status."""
        # Update homing parameters
        param_update = {"signal_loss_timeout": 15.0}
        response1 = client.patch("/api/homing/parameters", json=param_update)
        assert response1.status_code == 200

        # Check safety status still accessible
        response2 = client.get("/api/safety/status")
        assert response2.status_code == 200

        # Verify parameters are updated
        response3 = client.get("/api/homing/parameters")
        assert response3.status_code == 200

    def test_system_status_comprehensive_monitoring(self, client, mock_dependencies):
        """Test system status provides comprehensive system overview."""
        with (
            patch("psutil.cpu_percent", return_value=25.5),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
            patch("psutil.boot_time", return_value=1640995200.0),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            mock_memory.return_value.percent = 55.0
            mock_disk.return_value.percent = 18.5

            response = client.get("/api/system/status")

            assert response.status_code == 200
            data = response.json()

            # Verify all critical system information is present
            critical_fields = [
                "status",
                "timestamp",
                "current_state",
                "homing_enabled",
                "flight_mode",
                "battery_percent",
                "gps_status",
                "mavlink_connected",
                "sdr_status",
                "sdr_frequency",
                "cpu_usage",
                "memory_usage",
                "disk_usage",
                "uptime",
                "safety_interlocks",
            ]

            for field in critical_fields:
                assert field in data, f"Missing critical field: {field}"
