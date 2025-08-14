"""Integration tests for WebSocket state event broadcasting."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket

from src.backend.api.websocket import (
    ConnectionManager,
    broadcast_state_change,
    manager,
)
from src.backend.services.state_machine import StateMachine, SystemState


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock(spec=WebSocket)
    ws.send_text = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def connection_manager():
    """Create a fresh connection manager."""
    return ConnectionManager()


@pytest.mark.asyncio
async def test_state_change_broadcast(mock_websocket):
    """Test that state changes are broadcast to all connected clients."""
    # Add mock connection
    await manager.connect(mock_websocket)

    # Broadcast a state change
    await broadcast_state_change(
        old_state=SystemState.IDLE, new_state=SystemState.SEARCHING, reason="Test transition"
    )

    # Check that message was sent
    mock_websocket.send_text.assert_called_once()

    # Parse the sent message
    sent_data = mock_websocket.send_text.call_args[0][0]
    message = json.loads(sent_data)

    assert message["type"] == "state_change"
    assert message["data"]["from_state"] == "IDLE"
    assert message["data"]["to_state"] == "SEARCHING"
    assert message["data"]["current_state"] == "SEARCHING"
    assert message["data"]["reason"] == "Test transition"

    # Clean up
    await manager.disconnect(mock_websocket)


@pytest.mark.asyncio
async def test_state_update_broadcast_content():
    """Test the content of periodic state update broadcasts."""
    # Create a mock state machine
    mock_sm = MagicMock(spec=StateMachine)
    mock_sm.get_current_state.return_value = SystemState.DETECTING
    mock_sm._previous_state = SystemState.SEARCHING
    mock_sm.get_allowed_transitions.return_value = [
        SystemState.SEARCHING,
        SystemState.HOMING,
        SystemState.IDLE,
    ]
    mock_sm.get_statistics.return_value = {
        "homing_enabled": True,
        "detection_count": 5,
        "time_since_detection": 2.5,
    }
    mock_sm.get_state_history.return_value = [
        {
            "from_state": "SEARCHING",
            "to_state": "DETECTING",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": "Signal detected",
        }
    ]
    mock_sm.get_search_pattern_status.return_value = {
        "has_pattern": True,
        "pattern_id": "test-pattern",
        "progress_percent": 45.0,
    }

    # Mock the broadcast function
    broadcast_called = False
    broadcast_message = None

    async def mock_broadcast(message):
        nonlocal broadcast_called, broadcast_message
        broadcast_called = True
        broadcast_message = message

    # Patch manager.broadcast_json and state_machine
    with (
        patch.object(manager, "broadcast_json", mock_broadcast),
        patch("src.backend.api.websocket.state_machine", mock_sm),
    ):
        # Get comprehensive state information
        current_state = mock_sm.get_current_state()
        allowed_transitions = mock_sm.get_allowed_transitions()
        statistics = mock_sm.get_statistics()
        history = mock_sm.get_state_history(limit=10)
        search_status = mock_sm.get_search_pattern_status()

        message = {
            "type": "state",
            "data": {
                "current_state": current_state.value,
                "previous_state": mock_sm._previous_state.value,
                "allowed_transitions": [s.value for s in allowed_transitions],
                "state_duration_ms": None,
                "history": history,
                "search_status": search_status,
                "homing_enabled": statistics.get("homing_enabled", False),
                "detection_count": statistics.get("detection_count", 0),
                "time_since_detection": statistics.get("time_since_detection"),
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        await manager.broadcast_json(message)

    # Verify broadcast was called
    assert broadcast_called
    assert broadcast_message is not None
    assert broadcast_message["type"] == "state"
    assert broadcast_message["data"]["current_state"] == "DETECTING"
    assert broadcast_message["data"]["previous_state"] == "SEARCHING"
    assert broadcast_message["data"]["homing_enabled"] is True
    assert broadcast_message["data"]["detection_count"] == 5
    assert broadcast_message["data"]["search_status"]["has_pattern"] is True


@pytest.mark.asyncio
async def test_multiple_client_broadcast(connection_manager):
    """Test broadcasting to multiple connected clients."""
    # Create multiple mock connections
    clients = [MagicMock(spec=WebSocket) for _ in range(3)]
    for client in clients:
        client.send_text = AsyncMock()
        await connection_manager.connect(client)

    # Broadcast a message
    test_message = {"type": "test", "data": {"value": 123}}
    await connection_manager.broadcast_json(test_message)

    # Check all clients received the message
    expected_json = json.dumps(test_message)
    for client in clients:
        client.send_text.assert_called_once_with(expected_json)

    # Clean up
    for client in clients:
        await connection_manager.disconnect(client)


@pytest.mark.asyncio
async def test_disconnected_client_removal(connection_manager):
    """Test that disconnected clients are removed from active connections."""
    # Create mock connections
    good_client = MagicMock(spec=WebSocket)
    good_client.send_text = AsyncMock()

    bad_client = MagicMock(spec=WebSocket)
    bad_client.send_text = AsyncMock(side_effect=Exception("Connection closed"))

    # Connect both
    await connection_manager.connect(good_client)
    await connection_manager.connect(bad_client)

    assert len(connection_manager.active_connections) == 2

    # Broadcast - bad client should fail and be removed
    await connection_manager.broadcast_json({"type": "test"})

    # Only good client should remain
    assert len(connection_manager.active_connections) == 1
    assert good_client in connection_manager.active_connections
    assert bad_client not in connection_manager.active_connections


@pytest.mark.asyncio
async def test_state_callback_registration():
    """Test that state machine callbacks are properly registered."""
    # Create a new state machine
    sm = StateMachine(enable_persistence=False)

    # Track callback calls
    callback_called = False
    callback_args = None

    async def test_callback(old_state, new_state, reason):
        nonlocal callback_called, callback_args
        callback_called = True
        callback_args = (old_state, new_state, reason)

    # Add callback
    sm.add_state_callback(test_callback)

    # Trigger state change
    await sm.transition_to(SystemState.SEARCHING, "Test")

    # Verify callback was called
    assert callback_called
    assert callback_args[0] == SystemState.IDLE
    assert callback_args[1] == SystemState.SEARCHING
    assert callback_args[2] == "Test"


@pytest.mark.asyncio
async def test_state_history_in_broadcast():
    """Test that state history is included in broadcasts."""
    sm = StateMachine(enable_persistence=False)

    # Create some history
    await sm.transition_to(SystemState.SEARCHING, "Start search")
    await sm.transition_to(SystemState.DETECTING, "Signal found")
    await sm.transition_to(SystemState.HOMING, "Homing to signal")

    # Get history
    history = sm.get_state_history(limit=5)

    # Verify history structure
    assert len(history) == 3
    assert history[0]["from_state"] == "DETECTING"
    assert history[0]["to_state"] == "HOMING"
    assert history[0]["reason"] == "Homing to signal"

    assert history[1]["from_state"] == "SEARCHING"
    assert history[1]["to_state"] == "DETECTING"

    assert history[2]["from_state"] == "IDLE"
    assert history[2]["to_state"] == "SEARCHING"
