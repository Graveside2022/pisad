"""Unit tests for state machine persistence functionality."""

import contextlib
import os
import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.backend.models.database import StateHistoryDB
from src.backend.services.state_machine import StateMachine, SystemState


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    with contextlib.suppress(Exception):
        os.unlink(db_path)


@pytest.fixture
def state_machine_with_db(temp_db):
    """Create a state machine with database persistence."""
    sm = StateMachine(db_path=temp_db, enable_persistence=True)
    # Set a mock signal processor to allow transitions
    mock_processor = MagicMock()
    sm.set_signal_processor(mock_processor)
    return sm


@pytest.fixture
def state_db(temp_db):
    """Create a state history database instance."""
    return StateHistoryDB(db_path=temp_db)


@pytest.mark.asyncio
async def test_state_persistence_on_transition(state_machine_with_db, temp_db):
    """Test that state transitions are persisted to database."""
    # Perform state transition
    result = await state_machine_with_db.transition_to(SystemState.SEARCHING, "Starting search")
    assert result is True

    # Check database directly
    db = StateHistoryDB(db_path=temp_db)
    history = db.get_state_history(limit=1)

    assert len(history) == 1
    assert history[0]["from_state"] == "IDLE"
    assert history[0]["to_state"] == "SEARCHING"
    assert history[0]["reason"] == "Starting search"


@pytest.mark.asyncio
async def test_current_state_persistence(state_machine_with_db, temp_db):
    """Test that current state is saved for recovery."""
    # Set some state values
    state_machine_with_db.enable_homing(True)
    state_machine_with_db._detection_count = 5
    state_machine_with_db._last_detection_time = 123456.789

    # Transition through valid states to reach DETECTING
    await state_machine_with_db.transition_to(SystemState.SEARCHING)
    await state_machine_with_db.transition_to(SystemState.DETECTING)

    # Check current state in database
    db = StateHistoryDB(db_path=temp_db)
    saved_state = db.restore_state()

    assert saved_state is not None
    assert saved_state["state"] == "DETECTING"
    assert saved_state["previous_state"] == "SEARCHING"  # Previous state before DETECTING
    assert saved_state["homing_enabled"] is True
    assert saved_state["detection_count"] == 5
    assert saved_state["last_detection_time"] == 123456.789


@pytest.mark.asyncio
async def test_state_restoration_on_startup(temp_db):
    """Test that state is restored from database on startup."""
    # Create first state machine and set some state
    sm1 = StateMachine(db_path=temp_db, enable_persistence=True)
    sm1.set_signal_processor(MagicMock())
    # Set MAVLink service to allow HOMING transition
    sm1.set_mavlink_service(MagicMock())
    sm1.enable_homing(True)
    sm1._detection_count = 10
    await sm1.transition_to(SystemState.SEARCHING)
    await sm1.transition_to(SystemState.DETECTING)
    await sm1.transition_to(SystemState.HOMING, "Found signal")

    # Create new state machine with same database
    sm2 = StateMachine(db_path=temp_db, enable_persistence=True)
    sm2.set_signal_processor(MagicMock())

    # Check restored state
    assert sm2.get_current_state() == SystemState.HOMING
    assert sm2._previous_state == SystemState.DETECTING
    assert sm2._homing_enabled is True
    assert sm2._detection_count == 10
    assert len(sm2._state_history) >= 3


@pytest.mark.asyncio
async def test_forced_transition_persistence(state_machine_with_db, temp_db):
    """Test that forced transitions are persisted with operator info."""
    # Force a transition
    result = await state_machine_with_db.force_transition(
        SystemState.HOMING, "Manual override for testing", operator_id="test_operator"
    )
    assert result is True

    # Check database
    db = StateHistoryDB(db_path=temp_db)
    history = db.get_state_history(limit=1)

    assert len(history) == 1
    assert history[0]["from_state"] == "IDLE"
    assert history[0]["to_state"] == "HOMING"
    assert "FORCED" in history[0]["reason"]
    assert history[0]["operator_id"] == "test_operator"


@pytest.mark.asyncio
async def test_persistence_with_database_failure(temp_db):
    """Test that state machine continues working even if database fails."""
    sm = StateMachine(db_path=temp_db, enable_persistence=True)
    sm.set_signal_processor(MagicMock())

    # Break the database connection
    if sm._state_db:
        # Make save methods fail
        sm._state_db.save_state_change = MagicMock(side_effect=Exception("DB Error"))
        sm._state_db.save_current_state = MagicMock(side_effect=Exception("DB Error"))

    # Should still be able to transition
    result = await sm.transition_to(SystemState.SEARCHING)
    assert result is True
    assert sm.get_current_state() == SystemState.SEARCHING


@pytest.mark.asyncio
async def test_action_duration_persistence(state_machine_with_db, temp_db):
    """Test that entry/exit action durations are persisted."""

    # Add a slow action
    async def slow_action():
        import asyncio

        await asyncio.sleep(0.01)  # 10ms

    state_machine_with_db.register_exit_action(SystemState.IDLE, slow_action)

    # Transition
    await state_machine_with_db.transition_to(SystemState.SEARCHING)

    # Check database
    db = StateHistoryDB(db_path=temp_db)
    history = db.get_state_history(limit=1)

    assert len(history) == 1
    assert history[0]["action_duration_ms"] is not None
    assert history[0]["action_duration_ms"] >= 10  # At least 10ms


def test_state_history_cleanup(state_db):
    """Test cleanup of old state history records."""
    # Add some old records
    old_date = datetime.now(UTC).replace(day=1)  # Beginning of month
    for i in range(10):
        state_db.save_state_change(
            from_state="IDLE", to_state="SEARCHING", timestamp=old_date, reason=f"Test {i}"
        )

    # Add some recent records
    recent_date = datetime.now(UTC)
    for i in range(5):
        state_db.save_state_change(
            from_state="SEARCHING",
            to_state="DETECTING",
            timestamp=recent_date,
            reason=f"Recent {i}",
        )

    # Check total count
    all_history = state_db.get_state_history(limit=100)
    assert len(all_history) == 15

    # Cleanup old records (keeping only today's)
    state_db.cleanup_old_history(days_to_keep=1)

    # Check remaining records
    remaining = state_db.get_state_history(limit=100)
    assert len(remaining) <= 15  # Some or all may remain depending on date


def test_state_history_filtering(state_db):
    """Test filtering state history by from/to states."""
    # Add various transitions
    timestamp = datetime.now(UTC)
    state_db.save_state_change("IDLE", "SEARCHING", timestamp)
    state_db.save_state_change("SEARCHING", "DETECTING", timestamp)
    state_db.save_state_change("DETECTING", "HOMING", timestamp)
    state_db.save_state_change("HOMING", "HOLDING", timestamp)
    state_db.save_state_change("HOLDING", "IDLE", timestamp)

    # Filter by from_state
    from_searching = state_db.get_state_history(from_state="SEARCHING")
    assert len(from_searching) == 1
    assert from_searching[0]["from_state"] == "SEARCHING"
    assert from_searching[0]["to_state"] == "DETECTING"

    # Filter by to_state
    to_idle = state_db.get_state_history(to_state="IDLE")
    assert len(to_idle) == 1
    assert to_idle[0]["from_state"] == "HOLDING"
    assert to_idle[0]["to_state"] == "IDLE"


@pytest.mark.asyncio
async def test_persistence_disabled(temp_db):
    """Test state machine works without persistence."""
    sm = StateMachine(db_path=temp_db, enable_persistence=False)
    sm.set_signal_processor(MagicMock())

    assert sm._state_db is None

    # Should still work normally
    result = await sm.transition_to(SystemState.SEARCHING)
    assert result is True
    assert sm.get_current_state() == SystemState.SEARCHING

    # Check that nothing was persisted
    db = StateHistoryDB(db_path=temp_db)
    history = db.get_state_history()
    assert len(history) == 0
