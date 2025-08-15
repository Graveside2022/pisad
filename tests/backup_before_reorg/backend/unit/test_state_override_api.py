"""Unit tests for state override API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.backend.api.routes.system import router
from src.backend.services.state_machine import StateMachine, SystemState


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_state_machine():
    """Create a mock state machine."""
    mock_sm = MagicMock(spec=StateMachine)
    mock_sm.get_current_state.return_value = SystemState.IDLE
    mock_sm._previous_state = SystemState.IDLE
    mock_sm.get_allowed_transitions.return_value = [SystemState.SEARCHING]
    mock_sm.force_transition = AsyncMock(return_value=True)
    mock_sm.get_statistics.return_value = {
        "current_state": "IDLE",
        "previous_state": "IDLE",
        "homing_enabled": False,
        "detection_count": 0,
        "state_changes": 0,
    }
    mock_sm.get_state_history.return_value = []
    mock_sm.get_search_pattern_status.return_value = {"has_pattern": False}
    mock_sm._state_db = None
    return mock_sm


def test_state_override_success(test_client, mock_state_machine, monkeypatch):
    """Test successful state override."""
    # Set up the mock state machine
    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_state_machine)

    # Get current date for token
    token = "override-" + datetime.now(UTC).strftime("%Y%m%d")

    response = test_client.post(
        "/system/state-override",
        json={
            "target_state": "SEARCHING",
            "reason": "Testing state override",
            "confirmation_token": token,
            "operator_id": "test_operator",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["operator_id"] == "test_operator"
    assert "new_state" in data
    assert "allowed_transitions" in data


def test_state_override_invalid_token(test_client, mock_state_machine, monkeypatch):
    """Test state override with invalid token."""
    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_state_machine)

    response = test_client.post(
        "/system/state-override",
        json={
            "target_state": "SEARCHING",
            "reason": "Testing",
            "confirmation_token": "invalid-token",
            "operator_id": "test_operator",
        },
    )

    assert response.status_code == 403
    data = response.json()
    assert "Invalid confirmation token" in data["detail"]


def test_state_override_invalid_state(test_client, mock_state_machine, monkeypatch):
    """Test state override with invalid state name."""
    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_state_machine)

    token = "override-" + datetime.now(UTC).strftime("%Y%m%d")

    response = test_client.post(
        "/system/state-override",
        json={
            "target_state": "INVALID_STATE",
            "reason": "Testing",
            "confirmation_token": token,
            "operator_id": "test_operator",
        },
    )

    assert response.status_code == 422  # Validation error


def test_state_override_no_state_machine(test_client):
    """Test state override when state machine is not initialized."""
    response = test_client.post(
        "/system/state-override",
        json={
            "target_state": "SEARCHING",
            "reason": "Testing",
            "confirmation_token": "override-20240101",
            "operator_id": "test_operator",
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "State machine not initialized" in data["detail"]


def test_get_current_state(test_client, mock_state_machine, monkeypatch):
    """Test getting current state information."""
    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_state_machine)

    response = test_client.get("/system/state")

    assert response.status_code == 200
    data = response.json()
    assert "current_state" in data
    assert "previous_state" in data
    assert "allowed_transitions" in data
    assert "statistics" in data
    assert "history" in data
    assert "search_status" in data


def test_get_state_history(test_client, mock_state_machine, monkeypatch):
    """Test getting state history."""
    # Set up mock history
    mock_state_machine.get_state_history.return_value = [
        {
            "from_state": "IDLE",
            "to_state": "SEARCHING",
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "Test transition",
        }
    ]

    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_state_machine)

    response = test_client.get("/system/state-history?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert "count" in data
    assert data["count"] == 1
    assert data["history"][0]["from_state"] == "IDLE"
    assert data["history"][0]["to_state"] == "SEARCHING"


def test_get_state_history_with_filters(test_client, mock_state_machine, monkeypatch):
    """Test getting state history with filters."""
    mock_state_machine.get_state_history.return_value = [
        {
            "from_state": "SEARCHING",
            "to_state": "DETECTING",
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "Signal found",
        }
    ]

    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_state_machine)

    response = test_client.get("/system/state-history?from_state=SEARCHING&to_state=DETECTING")

    assert response.status_code == 200
    data = response.json()
    assert data["filters"]["from_state"] == "SEARCHING"
    assert data["filters"]["to_state"] == "DETECTING"


def test_state_override_transition_failure(test_client, mock_state_machine, monkeypatch):
    """Test state override when transition fails."""
    mock_state_machine.force_transition = AsyncMock(return_value=False)
    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_state_machine)

    token = "override-" + datetime.now(UTC).strftime("%Y%m%d")

    response = test_client.post(
        "/system/state-override",
        json={
            "target_state": "HOMING",
            "reason": "Testing failed transition",
            "confirmation_token": token,
            "operator_id": "test_operator",
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "Failed to transition" in data["detail"]


def test_state_history_with_database(test_client, monkeypatch):
    """Test state history retrieval from database."""
    mock_sm = MagicMock(spec=StateMachine)
    mock_db = MagicMock()
    mock_db.get_state_history.return_value = [
        {
            "id": 1,
            "from_state": "IDLE",
            "to_state": "SEARCHING",
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "Database test",
            "operator_id": None,
            "action_duration_ms": 50,
            "created_at": datetime.now(UTC).isoformat(),
        }
    ]
    mock_sm._state_db = mock_db

    monkeypatch.setattr("src.backend.api.routes.system.state_machine", mock_sm)

    response = test_client.get("/system/state-history")

    assert response.status_code == 200
    data = response.json()
    assert len(data["history"]) == 1
    assert data["history"][0]["reason"] == "Database test"
    assert data["history"][0]["action_duration_ms"] == 50
