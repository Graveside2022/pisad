"""Comprehensive tests for state API routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.backend.services.state_machine import SystemState


@pytest.fixture
def mock_state_machine():
    """Create mock state machine for testing."""
    mock = MagicMock()
    mock.get_current_state.return_value = SystemState.IDLE
    mock._previous_state = SystemState.IDLE
    mock.get_allowed_transitions.return_value = [SystemState.SEARCHING, SystemState.DETECTING]
    mock._homing_enabled = False
    mock.get_state_duration.return_value = 10.5
    mock._state_timeouts = {SystemState.IDLE: 0}
    mock.get_state_history.return_value = []
    mock._state_history = []
    mock.get_statistics.return_value = {
        "current_state": "IDLE",
        "previous_state": "IDLE",
        "homing_enabled": False,
        "detection_count": 0,
        "last_detection_time": None,
        "time_since_detection": None,
        "state_changes": 0,
        "state_duration_seconds": 10.5,
        "state_timeout_seconds": 0
    }
    mock.get_telemetry_metrics.return_value = {
        "total_transitions": 0,
        "state_durations": {},
        "transition_frequencies": {},
        "average_transition_time_ms": 0.0,
        "current_state_duration_s": 10.5,
        "uptime_seconds": 100.0,
        "state_entry_counts": {}
    }
    mock.get_search_pattern_status.return_value = {
        "active": False,
        "paused": False,
        "progress": 0
    }
    mock.transition_to = AsyncMock(return_value=True)
    mock.force_transition = AsyncMock(return_value=True)
    mock.emergency_stop = AsyncMock()
    mock.send_telemetry_update = AsyncMock()
    mock.start_search_pattern = AsyncMock(return_value=True)
    mock.pause_search_pattern = AsyncMock(return_value=True)
    mock.resume_search_pattern = AsyncMock(return_value=True)
    mock.stop_search_pattern = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def test_client(mock_state_machine):
    """Create test client with mocked state machine."""
    # Create a minimal app for testing just the state routes
    from fastapi import FastAPI
    from src.backend.api.routes import state
    
    app = FastAPI()
    
    # Override the dependency directly
    def override_get_state_machine():
        return mock_state_machine
    
    app.dependency_overrides[state.get_state_machine] = override_get_state_machine
    app.include_router(state.router)
    
    with TestClient(app) as client:
        yield client


class TestGetCurrentState:
    """Tests for GET /api/state/current endpoint."""

    def test_get_current_state_success(self, test_client, mock_state_machine):
        """Test successful retrieval of current state."""
        response = test_client.get("/api/state/current")
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "IDLE"
        assert data["previous_state"] == "IDLE"
        assert "SEARCHING" in data["allowed_transitions"]
        assert "DETECTING" in data["allowed_transitions"]
        assert data["homing_enabled"] is False
        assert data["state_duration_seconds"] == 10.5
        assert data["state_timeout_seconds"] == 0

    def test_get_current_state_with_homing_enabled(self, test_client, mock_state_machine):
        """Test current state when homing is enabled."""
        mock_state_machine._homing_enabled = True
        response = test_client.get("/api/state/current")
        assert response.status_code == 200
        assert response.json()["homing_enabled"] is True

    def test_get_current_state_exception(self, test_client, mock_state_machine):
        """Test error handling when getting current state fails."""
        mock_state_machine.get_current_state.side_effect = Exception("Test error")
        response = test_client.get("/api/state/current")
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]


class TestTransitionState:
    """Tests for POST /api/state/transition endpoint."""

    @pytest.mark.asyncio
    async def test_transition_success(self, test_client, mock_state_machine):
        """Test successful state transition."""
        response = test_client.post(
            "/api/state/transition",
            json={"target_state": "searching", "reason": "Test transition"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Transitioned to SEARCHING" in data["message"]
        mock_state_machine.transition_to.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_invalid_state(self, test_client):
        """Test transition with invalid state name."""
        response = test_client.post(
            "/api/state/transition",
            json={"target_state": "INVALID_STATE", "reason": "Test"}
        )
        assert response.status_code == 400
        assert "Invalid state" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_transition_not_allowed(self, test_client, mock_state_machine):
        """Test transition that is not allowed."""
        mock_state_machine.transition_to.return_value = False
        response = test_client.post(
            "/api/state/transition",
            json={"target_state": "homing", "reason": "Test"}
        )
        assert response.status_code == 400
        assert "Invalid transition" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_transition_exception(self, test_client, mock_state_machine):
        """Test error handling during transition."""
        mock_state_machine.transition_to.side_effect = Exception("Transition failed")
        response = test_client.post(
            "/api/state/transition",
            json={"target_state": "searching", "reason": "Test"}
        )
        assert response.status_code == 500
        assert "Transition failed" in response.json()["detail"]


class TestForceTransition:
    """Tests for POST /api/state/force-transition endpoint."""

    @pytest.mark.asyncio
    async def test_force_transition_success(self, test_client, mock_state_machine):
        """Test successful forced state transition."""
        response = test_client.post(
            "/api/state/force-transition",
            json={
                "target_state": "homing",
                "reason": "Emergency override",
                "operator_id": "test_operator"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Forced transition to HOMING" in data["message"]
        assert "forced transition" in data["warning"].lower()
        mock_state_machine.force_transition.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_transition_invalid_state(self, test_client):
        """Test force transition with invalid state."""
        response = test_client.post(
            "/api/state/force-transition",
            json={
                "target_state": "INVALID",
                "reason": "Test",
                "operator_id": "test"
            }
        )
        assert response.status_code == 400
        assert "Invalid state" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_force_transition_exception(self, test_client, mock_state_machine):
        """Test error handling during forced transition."""
        mock_state_machine.force_transition.side_effect = Exception("Force failed")
        response = test_client.post(
            "/api/state/force-transition",
            json={
                "target_state": "idle",
                "reason": "Test",
                "operator_id": "test"
            }
        )
        assert response.status_code == 500
        assert "Force failed" in response.json()["detail"]


class TestEmergencyStop:
    """Tests for POST /api/state/emergency-stop endpoint."""

    @pytest.mark.asyncio
    async def test_emergency_stop_success(self, test_client, mock_state_machine):
        """Test successful emergency stop."""
        response = test_client.post("/api/state/emergency-stop")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["current_state"] == "IDLE"
        assert "Emergency stop executed" in data["message"]
        mock_state_machine.emergency_stop.assert_called_once_with("API emergency stop request")

    @pytest.mark.asyncio
    async def test_emergency_stop_exception(self, test_client, mock_state_machine):
        """Test error handling during emergency stop."""
        mock_state_machine.emergency_stop.side_effect = Exception("Stop failed")
        response = test_client.post("/api/state/emergency-stop")
        assert response.status_code == 500
        assert "Stop failed" in response.json()["detail"]


class TestHomingControl:
    """Tests for PUT /api/state/homing/{enabled} endpoint."""

    def test_enable_homing_success(self, test_client, mock_state_machine):
        """Test enabling homing."""
        response = test_client.put("/api/state/homing/true")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["homing_enabled"] is True
        assert "Homing enabled" in data["message"]
        mock_state_machine.enable_homing.assert_called_once_with(True)

    def test_disable_homing_success(self, test_client, mock_state_machine):
        """Test disabling homing."""
        response = test_client.put("/api/state/homing/false")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["homing_enabled"] is False
        assert "Homing disabled" in data["message"]
        mock_state_machine.enable_homing.assert_called_once_with(False)

    def test_homing_control_exception(self, test_client, mock_state_machine):
        """Test error handling when setting homing state."""
        mock_state_machine.enable_homing.side_effect = Exception("Homing error")
        response = test_client.put("/api/state/homing/true")
        assert response.status_code == 500
        assert "Homing error" in response.json()["detail"]


class TestStateTimeout:
    """Tests for PUT /api/state/timeout endpoint."""

    def test_set_timeout_success(self, test_client, mock_state_machine):
        """Test setting state timeout."""
        response = test_client.put(
            "/api/state/timeout",
            json={"state": "searching", "timeout_seconds": 30.0}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["state"] == "SEARCHING"
        assert data["timeout_seconds"] == 30.0
        mock_state_machine.set_state_timeout.assert_called_once()

    def test_set_timeout_zero_to_disable(self, test_client, mock_state_machine):
        """Test disabling timeout by setting to zero."""
        response = test_client.put(
            "/api/state/timeout",
            json={"state": "idle", "timeout_seconds": 0}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["timeout_seconds"] == 0

    def test_set_timeout_invalid_state(self, test_client):
        """Test setting timeout for invalid state."""
        response = test_client.put(
            "/api/state/timeout",
            json={"state": "INVALID", "timeout_seconds": 10}
        )
        assert response.status_code == 400
        assert "Invalid state" in response.json()["detail"]

    def test_set_timeout_exception(self, test_client, mock_state_machine):
        """Test error handling when setting timeout."""
        mock_state_machine.set_state_timeout.side_effect = Exception("Timeout error")
        response = test_client.put(
            "/api/state/timeout",
            json={"state": "idle", "timeout_seconds": 10}
        )
        assert response.status_code == 500
        assert "Timeout error" in response.json()["detail"]


class TestStateHistory:
    """Tests for GET /api/state/history endpoint."""

    def test_get_history_default_limit(self, test_client, mock_state_machine):
        """Test getting state history with default limit."""
        mock_state_machine.get_state_history.return_value = [
            {"state": "IDLE", "timestamp": 1234567890},
            {"state": "ARMING", "timestamp": 1234567900}
        ]
        mock_state_machine._state_history = ["event1", "event2"]
        
        response = test_client.get("/api/state/history")
        assert response.status_code == 200
        data = response.json()
        assert len(data["history"]) == 2
        assert data["total_transitions"] == 2

    def test_get_history_custom_limit(self, test_client, mock_state_machine):
        """Test getting state history with custom limit."""
        mock_state_machine.get_state_history.return_value = []
        response = test_client.get("/api/state/history?limit=5")
        assert response.status_code == 200
        mock_state_machine.get_state_history.assert_called_once_with(limit=5)

    def test_get_history_all_events(self, test_client, mock_state_machine):
        """Test getting all history events."""
        response = test_client.get("/api/state/history?limit=0")
        assert response.status_code == 200
        mock_state_machine.get_state_history.assert_called_once_with(limit=0)

    def test_get_history_exception(self, test_client, mock_state_machine):
        """Test error handling when getting history."""
        mock_state_machine.get_state_history.side_effect = Exception("History error")
        response = test_client.get("/api/state/history")
        assert response.status_code == 500
        assert "History error" in response.json()["detail"]


class TestStateStatistics:
    """Tests for GET /api/state/statistics endpoint."""

    def test_get_statistics_success(self, test_client, mock_state_machine):
        """Test getting state statistics."""
        response = test_client.get("/api/state/statistics")
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "IDLE"
        assert data["homing_enabled"] is False
        assert data["detection_count"] == 0
        assert data["state_duration_seconds"] == 10.5

    def test_get_statistics_with_detections(self, test_client, mock_state_machine):
        """Test statistics with detection data."""
        mock_state_machine.get_statistics.return_value = {
            "current_state": "SEARCHING",
            "previous_state": "IDLE",
            "homing_enabled": True,
            "detection_count": 5,
            "last_detection_time": 1234567890.0,
            "time_since_detection": 15.5,
            "state_changes": 10,
            "state_duration_seconds": 30.0,
            "state_timeout_seconds": 60.0
        }
        response = test_client.get("/api/state/statistics")
        assert response.status_code == 200
        data = response.json()
        assert data["detection_count"] == 5
        assert data["time_since_detection"] == 15.5

    def test_get_statistics_exception(self, test_client, mock_state_machine):
        """Test error handling when getting statistics."""
        mock_state_machine.get_statistics.side_effect = Exception("Stats error")
        response = test_client.get("/api/state/statistics")
        assert response.status_code == 500
        assert "Stats error" in response.json()["detail"]


class TestTelemetryMetrics:
    """Tests for GET /api/state/telemetry endpoint."""

    def test_get_telemetry_success(self, test_client, mock_state_machine):
        """Test getting telemetry metrics."""
        response = test_client.get("/api/state/telemetry")
        assert response.status_code == 200
        data = response.json()
        assert data["total_transitions"] == 0
        assert data["current_state_duration_s"] == 10.5
        assert data["uptime_seconds"] == 100.0

    def test_get_telemetry_with_data(self, test_client, mock_state_machine):
        """Test telemetry with actual metrics."""
        mock_state_machine.get_telemetry_metrics.return_value = {
            "total_transitions": 50,
            "state_durations": {"IDLE": 100.0, "SEARCHING": 200.0},
            "transition_frequencies": {"IDLE_TO_SEARCHING": 10},
            "average_transition_time_ms": 150.5,
            "current_state_duration_s": 45.0,
            "uptime_seconds": 1000.0,
            "state_entry_counts": {"IDLE": 20, "SEARCHING": 15}
        }
        response = test_client.get("/api/state/telemetry")
        assert response.status_code == 200
        data = response.json()
        assert data["total_transitions"] == 50
        assert data["state_durations"]["SEARCHING"] == 200.0
        assert data["average_transition_time_ms"] == 150.5

    def test_get_telemetry_exception(self, test_client, mock_state_machine):
        """Test error handling when getting telemetry."""
        mock_state_machine.get_telemetry_metrics.side_effect = Exception("Telemetry error")
        response = test_client.get("/api/state/telemetry")
        assert response.status_code == 500
        assert "Telemetry error" in response.json()["detail"]


class TestTelemetryUpdate:
    """Tests for POST /api/state/telemetry/send endpoint."""

    @pytest.mark.asyncio
    async def test_send_telemetry_success(self, test_client, mock_state_machine):
        """Test sending telemetry update."""
        response = test_client.post("/api/state/telemetry/send")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Telemetry update sent" in data["message"]
        mock_state_machine.send_telemetry_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_telemetry_exception(self, test_client, mock_state_machine):
        """Test error handling when sending telemetry."""
        mock_state_machine.send_telemetry_update.side_effect = Exception("Send failed")
        response = test_client.post("/api/state/telemetry/send")
        assert response.status_code == 500
        assert "Send failed" in response.json()["detail"]


class TestSearchPatternStatus:
    """Tests for GET /api/state/search-pattern/status endpoint."""

    def test_get_pattern_status_inactive(self, test_client, mock_state_machine):
        """Test getting search pattern status when inactive."""
        response = test_client.get("/api/state/search-pattern/status")
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False
        assert data["paused"] is False
        assert data["progress"] == 0

    def test_get_pattern_status_active(self, test_client, mock_state_machine):
        """Test getting search pattern status when active."""
        mock_state_machine.get_search_pattern_status.return_value = {
            "active": True,
            "paused": False,
            "progress": 50,
            "current_waypoint": 5,
            "total_waypoints": 10
        }
        response = test_client.get("/api/state/search-pattern/status")
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is True
        assert data["progress"] == 50
        assert data["current_waypoint"] == 5

    def test_get_pattern_status_exception(self, test_client, mock_state_machine):
        """Test error handling when getting pattern status."""
        mock_state_machine.get_search_pattern_status.side_effect = Exception("Status error")
        response = test_client.get("/api/state/search-pattern/status")
        assert response.status_code == 500
        assert "Status error" in response.json()["detail"]


class TestSearchPatternControl:
    """Tests for search pattern control endpoints."""

    @pytest.mark.asyncio
    async def test_start_pattern_success(self, test_client, mock_state_machine):
        """Test starting search pattern."""
        response = test_client.post("/api/state/search-pattern/start")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Search pattern started" in data["message"]
        mock_state_machine.start_search_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_pattern_failure(self, test_client, mock_state_machine):
        """Test failed pattern start."""
        mock_state_machine.start_search_pattern.return_value = False
        response = test_client.post("/api/state/search-pattern/start")
        assert response.status_code == 400
        assert "Failed to start search pattern" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_pause_pattern_success(self, test_client, mock_state_machine):
        """Test pausing search pattern."""
        response = test_client.post("/api/state/search-pattern/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Search pattern paused" in data["message"]

    @pytest.mark.asyncio
    async def test_pause_pattern_not_executing(self, test_client, mock_state_machine):
        """Test pausing when pattern not executing."""
        mock_state_machine.pause_search_pattern.return_value = False
        response = test_client.post("/api/state/search-pattern/pause")
        assert response.status_code == 400
        assert "Cannot pause" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_resume_pattern_success(self, test_client, mock_state_machine):
        """Test resuming search pattern."""
        response = test_client.post("/api/state/search-pattern/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Search pattern resumed" in data["message"]

    @pytest.mark.asyncio
    async def test_resume_pattern_not_paused(self, test_client, mock_state_machine):
        """Test resuming when not paused."""
        mock_state_machine.resume_search_pattern.return_value = False
        response = test_client.post("/api/state/search-pattern/resume")
        assert response.status_code == 400
        assert "Cannot resume" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_stop_pattern_success(self, test_client, mock_state_machine):
        """Test stopping search pattern."""
        response = test_client.post("/api/state/search-pattern/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Search pattern stopped" in data["message"]

    @pytest.mark.asyncio
    async def test_stop_pattern_not_active(self, test_client, mock_state_machine):
        """Test stopping when no active pattern."""
        mock_state_machine.stop_search_pattern.return_value = False
        response = test_client.post("/api/state/search-pattern/stop")
        assert response.status_code == 400
        assert "No active search pattern" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_pattern_control_exceptions(self, test_client, mock_state_machine):
        """Test error handling in pattern control."""
        mock_state_machine.start_search_pattern.side_effect = Exception("Pattern error")
        response = test_client.post("/api/state/search-pattern/start")
        assert response.status_code == 500
        assert "Pattern error" in response.json()["detail"]