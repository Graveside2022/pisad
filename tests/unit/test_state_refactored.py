"""Unit tests for refactored state machine components."""

import pytest
from datetime import datetime

from src.backend.services.state import (
    StateChangeEvent,

pytestmark = pytest.mark.serial
    StateEventHandler,
    StateHistory,
    StatePersistence,
    StateTransitionManager,
    StateValidator,
    SystemState,
)


class TestStateTransitionManager:
    """Test the core transition logic."""

    def test_initial_state(self):
        """Test initial state is IDLE."""
        manager = StateTransitionManager()
        assert manager.get_current_state() == SystemState.IDLE

    def test_valid_transition(self):
        """Test valid state transitions."""
        manager = StateTransitionManager()

        # IDLE -> SEARCHING is valid
        assert manager.can_transition(SystemState.SEARCHING)
        manager.transition(SystemState.SEARCHING)
        assert manager.get_current_state() == SystemState.SEARCHING

        # SEARCHING -> DETECTING is valid
        assert manager.can_transition(SystemState.DETECTING)
        manager.transition(SystemState.DETECTING)
        assert manager.get_current_state() == SystemState.DETECTING

    def test_invalid_transition(self):
        """Test invalid transitions raise error."""
        manager = StateTransitionManager()

        # IDLE -> HOMING is invalid
        assert not manager.can_transition(SystemState.HOMING)

        with pytest.raises(Exception):  # StateTransitionError
            manager.transition(SystemState.HOMING)

    def test_get_allowed_transitions(self):
        """Test getting allowed transitions."""
        manager = StateTransitionManager()

        # From IDLE
        allowed = manager.get_allowed_transitions()
        assert SystemState.SEARCHING in allowed
        assert SystemState.DETECTING in allowed
        assert SystemState.HOMING not in allowed

    def test_reset(self):
        """Test resetting to IDLE."""
        manager = StateTransitionManager()
        manager.transition(SystemState.SEARCHING)

        manager.reset()
        assert manager.get_current_state() == SystemState.IDLE
        assert manager.get_previous_state() == SystemState.SEARCHING


class TestStateValidator:
    """Test state validation logic."""

    def test_homing_enabled_check(self):
        """Test homing mode enable/disable."""
        validator = StateValidator()

        # Initially disabled
        assert not validator.is_homing_enabled()

        # Cannot transition to HOMING when disabled
        valid, reason = validator.validate_transition(
            SystemState.DETECTING, SystemState.HOMING
        )
        assert not valid
        assert "not enabled" in reason

        # Enable homing
        validator.enable_homing(True)
        assert validator.is_homing_enabled()

        # Now can transition to HOMING
        valid, reason = validator.validate_transition(
            SystemState.DETECTING, SystemState.HOMING
        )
        assert valid

    def test_guard_conditions(self):
        """Test guard condition registration and execution."""
        validator = StateValidator()

        # Register a guard that always fails
        def failing_guard():
            return False

        validator.register_guard(
            SystemState.IDLE, SystemState.SEARCHING, failing_guard
        )

        # Validation should fail
        valid, reason = validator.validate_transition(
            SystemState.IDLE, SystemState.SEARCHING
        )
        assert not valid
        assert "Guard condition failed" in reason

    def test_guard_count(self):
        """Test counting registered guards."""
        validator = StateValidator()
        assert validator.get_guard_count() == 0

        validator.register_guard(
            SystemState.IDLE, SystemState.SEARCHING, lambda: True
        )
        assert validator.get_guard_count() == 1


class TestStateEventHandler:
    """Test event handling logic."""

    @pytest.mark.asyncio
    async def test_entry_actions(self):
        """Test entry action execution."""
        handler = StateEventHandler()
        executed = []

        def entry_action(state):
            executed.append(f"entered_{state.value}")

        handler.register_entry_action(SystemState.SEARCHING, entry_action)
        await handler.execute_entry_actions(SystemState.SEARCHING)

        assert "entered_SEARCHING" in executed

    @pytest.mark.asyncio
    async def test_exit_actions(self):
        """Test exit action execution."""
        handler = StateEventHandler()
        executed = []

        def exit_action(state):
            executed.append(f"exited_{state.value}")

        handler.register_exit_action(SystemState.IDLE, exit_action)
        await handler.execute_exit_actions(SystemState.IDLE)

        assert "exited_IDLE" in executed

    @pytest.mark.asyncio
    async def test_callbacks(self):
        """Test state change callbacks."""
        handler = StateEventHandler()
        notifications = []

        def callback(change):
            notifications.append(change)

        handler.add_state_callback(callback)
        await handler.notify_callbacks(SystemState.IDLE, SystemState.SEARCHING)

        assert len(notifications) == 1
        assert notifications[0]["from"] == SystemState.IDLE
        assert notifications[0]["to"] == SystemState.SEARCHING

    def test_action_count(self):
        """Test counting registered actions."""
        handler = StateEventHandler()

        counts = handler.get_action_count()
        assert counts["total"] == 0

        handler.register_entry_action(SystemState.IDLE, lambda s: None)
        handler.register_exit_action(SystemState.IDLE, lambda s: None)
        handler.add_state_callback(lambda c: None)

        counts = handler.get_action_count()
        assert counts["entry_actions"] == 1
        assert counts["exit_actions"] == 1
        assert counts["callbacks"] == 1
        assert counts["total"] == 3


class TestStateHistory:
    """Test history tracking logic."""

    def test_record_transition(self):
        """Test recording state transitions."""
        history = StateHistory(max_history=10)

        event = StateChangeEvent(
            from_state=SystemState.IDLE,
            to_state=SystemState.SEARCHING,
            timestamp=datetime.now(),
            reason="Test transition"
        )

        history.record_transition(event)

        recent = history.get_recent_history(1)
        assert len(recent) == 1
        assert recent[0]["from_state"] == "IDLE"
        assert recent[0]["to_state"] == "SEARCHING"

    def test_statistics(self):
        """Test statistics calculation."""
        history = StateHistory()

        # Record some transitions
        for i in range(3):
            event = StateChangeEvent(
                from_state=SystemState.IDLE,
                to_state=SystemState.SEARCHING,
                timestamp=datetime.now()
            )
            history.record_transition(event)

        stats = history.get_statistics()
        assert stats["total_transitions"] == 3
        assert "SEARCHING" in stats["state_visit_counts"]
        assert stats["state_visit_counts"]["SEARCHING"] == 3

    def test_transition_count(self):
        """Test counting specific transitions."""
        history = StateHistory()

        # Record multiple transitions
        for _ in range(5):
            event = StateChangeEvent(
                from_state=SystemState.IDLE,
                to_state=SystemState.SEARCHING,
                timestamp=datetime.now()
            )
            history.record_transition(event)

        count = history.get_transition_count(
            SystemState.IDLE, SystemState.SEARCHING
        )
        assert count == 5


class TestStatePersistence:
    """Test database persistence logic."""

    def test_save_and_restore(self, tmp_path):
        """Test saving and restoring state."""
        db_path = tmp_path / "test.db"
        persistence = StatePersistence(str(db_path))

        # Save a state
        success = persistence.save_state(
            SystemState.SEARCHING,
            previous_state=SystemState.IDLE,
            reason="Test save"
        )
        assert success

        # Restore last state
        restored = persistence.restore_last_state()
        assert restored == SystemState.SEARCHING

    def test_disabled_persistence(self):
        """Test when persistence is disabled."""
        persistence = StatePersistence(enabled=False)

        # Should return None when disabled
        assert persistence.restore_last_state() is None

        # Save should succeed but do nothing
        assert persistence.save_state(SystemState.SEARCHING)

        # History should be empty
        assert persistence.get_state_history() == []

    def test_state_history(self, tmp_path):
        """Test retrieving state history."""
        db_path = tmp_path / "test.db"
        persistence = StatePersistence(str(db_path))

        # Save multiple states
        persistence.save_state(SystemState.IDLE)
        persistence.save_state(SystemState.SEARCHING, SystemState.IDLE, "Test")
        persistence.save_state(SystemState.DETECTING, SystemState.SEARCHING)

        # Get history
        history = persistence.get_state_history(2)
        assert len(history) == 2
        assert history[0]["state"] == "DETECTING"
        assert history[1]["state"] == "SEARCHING"
