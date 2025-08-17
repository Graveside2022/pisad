"""Unit tests for StateMachine service.

Tests the core state management functionality including transitions,
state persistence, and safety interlocks per PRD requirements.
"""

import asyncio
from datetime import datetime
from unittest.mock import patch

import pytest

from src.backend.services.state_machine import StateMachine, SystemState


class TestStateMachine:
    """Test state machine service."""

    @pytest.fixture
    def state_machine(self):
        """Provide StateMachine instance."""
        return StateMachine()

    def test_state_machine_initialization(self, state_machine):
        """Test StateMachine initializes correctly."""
        # State machine may restore from database, so we test properties exist
        assert state_machine.current_state in SystemState
        assert state_machine._last_detection_time is not None
        assert isinstance(state_machine._state_history, list)

        # Test transition to IDLE functionality - must be async
        # Remove this assertion since we can't await in sync test
        # Testing transition separately in async test

    @pytest.mark.asyncio
    async def test_transition_to_searching(self, state_machine):
        """Test transition from IDLE to SEARCHING state."""
        success = await state_machine.transition_to(SystemState.SEARCHING)

        assert success is True
        assert state_machine.current_state == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_transition_to_detecting(self, state_machine):
        """Test transition from SEARCHING to DETECTING state."""
        # First go to searching
        await state_machine.transition_to(SystemState.SEARCHING)

        # Then transition to detecting
        success = await state_machine.transition_to(SystemState.DETECTING)

        assert success is True
        assert state_machine.current_state == SystemState.DETECTING

    @pytest.mark.asyncio
    async def test_transition_to_homing(self, state_machine):
        """Test transition from DETECTING to HOMING state."""
        # Set up state chain: IDLE -> SEARCHING -> DETECTING -> HOMING
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)

        success = await state_machine.transition_to(SystemState.HOMING)

        assert success is True
        assert state_machine.current_state == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_transition_to_holding(self, state_machine):
        """Test transition from HOMING to HOLDING state."""
        # Set up state chain to HOMING
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)
        await state_machine.transition_to(SystemState.HOMING)

        success = await state_machine.transition_to(SystemState.HOLDING)

        assert success is True
        assert state_machine.current_state == SystemState.HOLDING

    @pytest.mark.asyncio
    async def test_invalid_transition_rejected(self, state_machine):
        """Test invalid state transitions are rejected."""
        # Cannot go directly from IDLE to HOMING
        success = await state_machine.transition_to(SystemState.HOMING)

        assert success is False
        assert state_machine.current_state == SystemState.IDLE

    def test_get_valid_transitions(self, state_machine):
        """Test getting valid transitions from current state."""
        # From IDLE, should only be able to go to SEARCHING
        valid_transitions = state_machine.get_valid_transitions()

        assert SystemState.SEARCHING in valid_transitions
        assert SystemState.HOMING not in valid_transitions

    def test_state_history_tracking(self, state_machine):
        """Test state transition history is tracked."""
        initial_history_length = len(state_machine._state_history)

        # Perform a valid transition
        asyncio.run(state_machine.transition_to(SystemState.SEARCHING))

        # History should have one more entry
        assert len(state_machine._state_history) == initial_history_length + 1

        # Latest entry should be the new state
        latest_entry = state_machine._state_history[-1]
        assert latest_entry["state"] == SystemState.SEARCHING
        assert isinstance(latest_entry["timestamp"], datetime)

    def test_get_state_history(self, state_machine):
        """Test state history retrieval."""
        # Perform several transitions
        asyncio.run(state_machine.transition_to(SystemState.SEARCHING))
        asyncio.run(state_machine.transition_to(SystemState.DETECTING))

        history = state_machine.get_state_history()

        assert isinstance(history, list)
        assert len(history) >= 2  # At least SEARCHING and DETECTING

        # Check history format
        for entry in history:
            assert "state" in entry
            assert "timestamp" in entry
            assert isinstance(entry["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_transition_with_safety_check(self, state_machine):
        """Test transitions respect safety interlocks."""
        # Mock safety check that fails
        with patch.object(state_machine, "_check_safety_conditions") as mock_safety:
            mock_safety.return_value = False

            success = await state_machine.transition_to(SystemState.HOMING)

            assert success is False
            assert state_machine.current_state == SystemState.IDLE

    def test_reset_to_idle(self, state_machine):
        """Test emergency reset to IDLE state."""
        # Go to some advanced state
        asyncio.run(state_machine.transition_to(SystemState.SEARCHING))
        asyncio.run(state_machine.transition_to(SystemState.DETECTING))

        # Reset should work from any state
        state_machine.reset_to_idle()

        assert state_machine.current_state == SystemState.IDLE

    def test_time_in_current_state(self, state_machine):
        """Test tracking time spent in current state."""
        import time

        # Transition to a new state
        asyncio.run(state_machine.transition_to(SystemState.SEARCHING))

        # Wait a brief moment
        time.sleep(0.1)

        time_in_state = state_machine.get_time_in_current_state()

        assert time_in_state > 0.05  # Should be at least 50ms
        assert time_in_state < 1.0  # But less than 1 second

    @pytest.mark.asyncio
    async def test_state_timeout_handling(self, state_machine):
        """Test automatic state timeout handling."""
        # Mock a state with timeout
        with patch.object(state_machine, "_get_state_timeout") as mock_timeout:
            mock_timeout.return_value = 0.1  # 100ms timeout

            await state_machine.transition_to(SystemState.DETECTING)

            # Wait for timeout
            await asyncio.sleep(0.2)

            # Should have automatically transitioned back
            # (exact behavior depends on implementation)
            assert state_machine.current_state != SystemState.DETECTING

    @pytest.mark.asyncio
    async def test_transition_callback_execution(self, state_machine):
        """Test state transition callbacks are executed."""
        callback_called = False

        async def test_callback(old_state, new_state):
            nonlocal callback_called
            callback_called = True

        state_machine.add_transition_callback(test_callback)

        await state_machine.transition_to(SystemState.SEARCHING)

        # This should fail initially as callback system needs implementation
        assert callback_called is True

    def test_state_validation_prevents_corruption(self, state_machine):
        """Test state validation prevents invalid state corruption."""
        # Try to set invalid state directly
        with pytest.raises(ValueError):
            state_machine._set_state("INVALID_STATE")

        # Should remain in valid state
        assert state_machine.current_state == SystemState.IDLE
