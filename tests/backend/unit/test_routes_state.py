"""Test suite for state machine API routes."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for state routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock state machine dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        mock_state_machine = Mock()
        mock_service_manager.state_machine = mock_state_machine

        yield {"state_machine": mock_state_machine, "service_manager": mock_service_manager}


class TestStateMachine:
    """Test state machine endpoints per PRD-FR3."""

    def test_get_current_state(self, client, mock_services):
        """Test current state retrieval per PRD-FR3."""
        mock_state = {
            "current_state": "SEARCHING",
            "previous_state": "IDLE",
            "state_duration": 120.5,
            "can_transition_to": ["HOMING", "IDLE"],
            "timestamp": "2025-08-17T15:00:00Z",
        }

        mock_services["state_machine"].get_current_state.return_value = mock_state
        response = client.get("/api/state/current")
        assert response.status_code == 200

    def test_request_state_transition(self, client, mock_services):
        """Test state transition request per PRD-FR3."""
        transition_request = {"target_state": "HOMING", "reason": "signal_detected"}

        mock_services["state_machine"].request_transition.return_value = {
            "status": "transition_accepted",
            "from_state": "SEARCHING",
            "to_state": "HOMING",
        }

        response = client.post("/api/state/transition", json=transition_request)
        assert response.status_code == 200

    def test_get_state_history(self, client, mock_services):
        """Test state history retrieval."""
        mock_history = {
            "transitions": [
                {"from": "IDLE", "to": "SEARCHING", "timestamp": "2025-08-17T15:00:00Z"},
                {"from": "SEARCHING", "to": "HOMING", "timestamp": "2025-08-17T15:05:00Z"},
            ],
            "total_transitions": 2,
        }

        mock_services["state_machine"].get_history.return_value = mock_history
        response = client.get("/api/state/history")
        assert response.status_code == 200

    def test_validate_state_transition(self, client, mock_services):
        """Test state transition validation."""
        validation_request = {"from_state": "SEARCHING", "to_state": "HOMING"}

        mock_services["state_machine"].validate_transition.return_value = {
            "valid": True,
            "conditions_met": ["signal_detected", "homing_enabled"],
        }

        response = client.post("/api/state/validate", json=validation_request)
        assert response.status_code == 200

    def test_get_state_machine_config(self, client, mock_services):
        """Test state machine configuration retrieval."""
        mock_config = {
            "states": ["IDLE", "SEARCHING", "HOMING", "RTL"],
            "transitions": {"IDLE": ["SEARCHING"], "SEARCHING": ["HOMING", "IDLE"]},
            "timeouts": {"SEARCHING": 600, "HOMING": 300},
        }

        mock_services["state_machine"].get_config.return_value = mock_config
        response = client.get("/api/state/config")
        assert response.status_code == 200
