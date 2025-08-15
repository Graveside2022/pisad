"""Comprehensive unit tests for enhanced state machine functionality."""

import asyncio
import contextlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.backend.services.state_machine import (
    SearchSubstate,

pytestmark = pytest.mark.serial
    StateMachine,
    SystemState,
)


@pytest.fixture
def state_machine():
    """Create a state machine without persistence for testing."""
    return StateMachine(enable_persistence=False)


@pytest.fixture
def state_machine_with_persistence():
    """Create a state machine with persistence for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    sm = StateMachine(db_path=db_path, enable_persistence=True)
    yield sm
    # Cleanup
    with contextlib.suppress(Exception):
        Path(db_path).unlink()


class TestTransitionGuards:
    """Test state transition guard conditions."""

    @pytest.mark.asyncio
    async def test_guard_prevents_searching_without_signal_processor(self, state_machine):
        """Test that SEARCHING requires signal processor."""
        # No signal processor set
        assert state_machine._signal_processor is None

        # Should fail to transition
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is False
        assert state_machine.get_current_state() == SystemState.IDLE

        # Set signal processor
        state_machine.set_signal_processor(MagicMock())

        # Should now succeed
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is True
        assert state_machine.get_current_state() == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_guard_prevents_homing_without_mavlink(self, state_machine):
        """Test that HOMING requires MAVLink service."""
        # Setup to get to DETECTING
        state_machine.set_signal_processor(MagicMock())
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)

        # Enable homing but no MAVLink
        state_machine.enable_homing(True)
        assert state_machine._mavlink_service is None

        # Should fail to transition
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is False
        assert state_machine.get_current_state() == SystemState.DETECTING

        # Set MAVLink service
        state_machine.set_mavlink_service(MagicMock())

        # Should now succeed
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is True
        assert state_machine.get_current_state() == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_guard_prevents_homing_when_disabled(self, state_machine):
        """Test that HOMING requires homing to be enabled."""
        # Setup
        state_machine.set_signal_processor(MagicMock())
        state_machine.set_mavlink_service(MagicMock())
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)

        # Homing disabled
        state_machine.enable_homing(False)

        # Should fail
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is False

        # Enable homing
        state_machine.enable_homing(True)

        # Should succeed
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is True


class TestStateTimeouts:
    """Test state timeout functionality."""

    @pytest.mark.asyncio
    async def test_state_timeout_triggers_transition(self, state_machine):
        """Test that state timeout triggers transition to IDLE."""
        # Set short timeout for SEARCHING
        state_machine.set_state_timeout(SystemState.SEARCHING, 0.1)
        state_machine.set_signal_processor(MagicMock())

        # Transition to SEARCHING
        await state_machine.transition_to(SystemState.SEARCHING)
        assert state_machine.get_current_state() == SystemState.SEARCHING

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should have transitioned to IDLE
        assert state_machine.get_current_state() == SystemState.IDLE

        # Check history for timeout reason
        history = state_machine.get_state_history(limit=1)
        assert "Timeout" in history[0]["reason"]

    @pytest.mark.asyncio
    async def test_state_change_cancels_timeout(self, state_machine):
        """Test that changing state cancels the timeout."""
        # Set timeout for SEARCHING
        state_machine.set_state_timeout(SystemState.SEARCHING, 0.2)
        state_machine.set_signal_processor(MagicMock())

        # Transition to SEARCHING
        await state_machine.transition_to(SystemState.SEARCHING)

        # Change state before timeout
        await asyncio.sleep(0.1)
        await state_machine.transition_to(SystemState.DETECTING)

        # Wait past original timeout
        await asyncio.sleep(0.15)

        # Should still be in DETECTING (timeout was cancelled)
        assert state_machine.get_current_state() == SystemState.DETECTING

    @pytest.mark.asyncio
    async def test_no_timeout_for_zero_duration(self, state_machine):
        """Test that zero timeout doesn't trigger transitions."""
        # IDLE has 0 timeout by default
        assert state_machine._state_timeouts[SystemState.IDLE] == 0

        # Stay in IDLE
        await asyncio.sleep(0.2)

        # Should still be IDLE
        assert state_machine.get_current_state() == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_get_state_duration(self, state_machine):
        """Test getting current state duration."""
        state_machine.set_signal_processor(MagicMock())

        # Transition to SEARCHING
        await state_machine.transition_to(SystemState.SEARCHING)

        # Wait a bit
        await asyncio.sleep(0.1)

        # Check duration
        duration = state_machine.get_state_duration()
        assert 0.09 < duration < 0.12

    @pytest.mark.asyncio
    async def test_force_transition_handles_timeout(self, state_machine):
        """Test that forced transitions properly handle timeouts."""
        # Set timeout for HOMING
        state_machine.set_state_timeout(SystemState.HOMING, 0.1)

        # Force transition to HOMING
        await state_machine.force_transition(SystemState.HOMING, "Test", "operator")

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should have timed out to IDLE
        assert state_machine.get_current_state() == SystemState.IDLE


class TestStateTransitions:
    """Test state transition logic and validation."""

    @pytest.mark.asyncio
    async def test_valid_transitions(self, state_machine):
        """Test all valid state transitions."""
        # Set required services for guards
        state_machine.set_signal_processor(MagicMock())
        state_machine.set_mavlink_service(MagicMock())
        state_machine.enable_homing(True)

        # IDLE -> SEARCHING
        assert await state_machine.transition_to(SystemState.SEARCHING)
        assert state_machine.get_current_state() == SystemState.SEARCHING

        # SEARCHING -> DETECTING
        assert await state_machine.transition_to(SystemState.DETECTING)
        assert state_machine.get_current_state() == SystemState.DETECTING

        # DETECTING -> HOMING
        assert await state_machine.transition_to(SystemState.HOMING)
        assert state_machine.get_current_state() == SystemState.HOMING

        # HOMING -> HOLDING
        assert await state_machine.transition_to(SystemState.HOLDING)
        assert state_machine.get_current_state() == SystemState.HOLDING

        # HOLDING -> IDLE
        assert await state_machine.transition_to(SystemState.IDLE)
        assert state_machine.get_current_state() == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_invalid_transitions(self, state_machine):
        """Test that invalid transitions are prevented."""
        # IDLE -> HOMING (invalid)
        assert not await state_machine.transition_to(SystemState.HOMING)
        assert state_machine.get_current_state() == SystemState.IDLE

        # IDLE -> DETECTING (invalid)
        assert not await state_machine.transition_to(SystemState.DETECTING)
        assert state_machine.get_current_state() == SystemState.IDLE

        # IDLE -> HOLDING (invalid)
        assert not await state_machine.transition_to(SystemState.HOLDING)
        assert state_machine.get_current_state() == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_transition_with_reason(self, state_machine):
        """Test state transitions with reasons are recorded."""
        # Set required service for guard
        state_machine.set_signal_processor(MagicMock())

        reason = "Test transition reason"
        await state_machine.transition_to(SystemState.SEARCHING, reason)

        history = state_machine.get_state_history(limit=1)
        assert len(history) == 1
        assert history[0]["reason"] == reason

    @pytest.mark.asyncio
    async def test_get_allowed_transitions(self, state_machine):
        """Test getting allowed transitions for each state."""
        # Set required services for guards
        state_machine.set_signal_processor(MagicMock())
        state_machine.set_mavlink_service(MagicMock())
        state_machine.enable_homing(True)

        # IDLE state
        allowed = state_machine.get_allowed_transitions()
        assert SystemState.SEARCHING in allowed
        assert len(allowed) == 1

        # Move to SEARCHING
        await state_machine.transition_to(SystemState.SEARCHING)
        allowed = state_machine.get_allowed_transitions()
        assert SystemState.IDLE in allowed
        assert SystemState.DETECTING in allowed

        # Move to DETECTING
        await state_machine.transition_to(SystemState.DETECTING)
        allowed = state_machine.get_allowed_transitions()
        assert SystemState.SEARCHING in allowed
        assert SystemState.HOMING in allowed
        assert SystemState.IDLE in allowed


class TestEntryExitActions:
    """Test entry and exit action functionality."""

    @pytest.mark.asyncio
    async def test_action_execution_tracking(self, state_machine):
        """Test that action execution is tracked and timed."""
        # Set required service for guard
        state_machine.set_signal_processor(MagicMock())

        execution_log = []

        async def track_entry():
            execution_log.append("entry")
            await asyncio.sleep(0.01)  # Simulate work

        async def track_exit():
            execution_log.append("exit")
            await asyncio.sleep(0.01)  # Simulate work

        # Clear default actions and add tracking actions
        state_machine._entry_actions[SystemState.SEARCHING] = []
        state_machine._exit_actions[SystemState.IDLE] = []
        state_machine.register_entry_action(SystemState.SEARCHING, track_entry)
        state_machine.register_exit_action(SystemState.IDLE, track_exit)

        # Transition and verify execution
        await state_machine.transition_to(SystemState.SEARCHING)

        assert execution_log == ["exit", "entry"]

    @pytest.mark.asyncio
    async def test_action_error_resilience(self, state_machine):
        """Test that action errors don't prevent transitions."""
        # Set required service for guard
        state_machine.set_signal_processor(MagicMock())

        async def failing_action():
            raise ValueError("Test error")

        state_machine.register_entry_action(SystemState.SEARCHING, failing_action)

        # Should still transition despite error
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is True
        assert state_machine.get_current_state() == SystemState.SEARCHING


class TestStatePersistence:
    """Test state persistence and recovery."""

    @pytest.mark.asyncio
    async def test_state_persistence_on_transition(self, state_machine_with_persistence):
        """Test that state transitions are persisted."""
        sm = state_machine_with_persistence

        # Set signal processor to satisfy guard condition
        sm.set_signal_processor(MagicMock())

        # Make transitions
        await sm.transition_to(SystemState.SEARCHING, "Test search")
        await sm.transition_to(SystemState.DETECTING, "Found signal")

        # Check persistence
        assert sm._state_db is not None
        history = sm._state_db.get_state_history(limit=2)
        assert len(history) >= 2

    @pytest.mark.asyncio
    async def test_state_recovery(self):
        """Test state recovery from database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create first state machine and make transitions
            sm1 = StateMachine(db_path=db_path, enable_persistence=True)
            sm1.set_signal_processor(MagicMock())  # Set required service
            sm1.enable_homing(True)
            await sm1.transition_to(SystemState.SEARCHING)
            await sm1.transition_to(SystemState.DETECTING)

            # Create second state machine with same DB
            sm2 = StateMachine(db_path=db_path, enable_persistence=True)

            # Should restore state
            assert sm2.get_current_state() == SystemState.DETECTING
            assert sm2._previous_state == SystemState.SEARCHING
            assert sm2._homing_enabled is True

        finally:
            Path(db_path).unlink()

    @pytest.mark.asyncio
    async def test_persistence_failure_handling(self, state_machine_with_persistence):
        """Test graceful handling of persistence failures."""
        sm = state_machine_with_persistence

        # Set signal processor to satisfy guard condition
        sm.set_signal_processor(MagicMock())

        # Break the database
        if sm._state_db:
            sm._state_db.save_state_change = MagicMock(side_effect=Exception("DB Error"))

        # Should still work
        result = await sm.transition_to(SystemState.SEARCHING)
        assert result is True


class TestForceTransition:
    """Test forced state transitions."""

    @pytest.mark.asyncio
    async def test_force_transition_bypasses_validation(self, state_machine):
        """Test that force_transition bypasses normal validation."""
        # Force invalid transition IDLE -> HOMING
        result = await state_machine.force_transition(
            SystemState.HOMING, "Test override", "test_operator"
        )

        assert result is True
        assert state_machine.get_current_state() == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_force_transition_logging(self, state_machine):
        """Test that forced transitions are logged with operator."""
        await state_machine.force_transition(SystemState.DETECTING, "Manual test", "operator123")

        history = state_machine.get_state_history(limit=1)
        assert "FORCED" in history[0]["reason"]
        assert "Manual test" in history[0]["reason"]

    @pytest.mark.asyncio
    async def test_force_transition_executes_actions(self, state_machine):
        """Test that forced transitions still execute entry/exit actions."""
        action_executed = False

        async def test_action():
            nonlocal action_executed
            action_executed = True

        state_machine.register_entry_action(SystemState.HOMING, test_action)

        await state_machine.force_transition(SystemState.HOMING, "Test", "operator")

        assert action_executed


class TestDetectionHandling:
    """Test signal detection event handling."""

    @pytest.mark.asyncio
    async def test_detection_triggers_transition(self, state_machine):
        """Test that detection triggers state transition."""
        # Set signal processor to satisfy guard condition
        state_machine.set_signal_processor(MagicMock())

        # Start searching
        await state_machine.transition_to(SystemState.SEARCHING)

        # Handle detection
        await state_machine.handle_detection(rssi=-50.0, confidence=85.0)

        # Should transition to DETECTING
        assert state_machine.get_current_state() == SystemState.DETECTING

    @pytest.mark.asyncio
    async def test_high_confidence_auto_homing(self, state_machine):
        """Test automatic transition to HOMING on high confidence."""
        # Set required services
        state_machine.set_signal_processor(MagicMock())
        state_machine.set_mavlink_service(MagicMock())
        state_machine.enable_homing(True)

        # Start searching and detect
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.handle_detection(rssi=-50.0, confidence=85.0)

        # Should auto-transition to HOMING
        assert state_machine.get_current_state() == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_signal_lost_handling(self, state_machine):
        """Test handling of signal loss."""
        # Set required services
        state_machine.set_signal_processor(MagicMock())
        state_machine.set_mavlink_service(MagicMock())
        state_machine.enable_homing(True)

        # Get to HOMING state
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)
        await state_machine.transition_to(SystemState.HOMING)

        # Handle signal loss
        await state_machine.handle_signal_lost()

        # Should return to SEARCHING
        assert state_machine.get_current_state() == SystemState.SEARCHING


class TestEmergencyStop:
    """Test emergency stop functionality."""

    @pytest.mark.asyncio
    async def test_emergency_stop_returns_to_idle(self, state_machine):
        """Test emergency stop returns system to IDLE."""
        # Set required services
        state_machine.set_signal_processor(MagicMock())
        state_machine.set_mavlink_service(MagicMock())
        state_machine.enable_homing(True)

        # Get to HOMING state
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)
        await state_machine.transition_to(SystemState.HOMING)

        # Emergency stop
        await state_machine.emergency_stop("Test emergency")

        # Should be in IDLE
        assert state_machine.get_current_state() == SystemState.IDLE
        assert not state_machine._homing_enabled

    @pytest.mark.asyncio
    async def test_emergency_stop_reason_logged(self, state_machine):
        """Test emergency stop reason is logged."""
        # First get out of IDLE state
        state_machine.set_signal_processor(MagicMock())
        await state_machine.transition_to(SystemState.SEARCHING)

        reason = "Critical failure detected"
        await state_machine.emergency_stop(reason)

        history = state_machine.get_state_history(limit=1)
        assert len(history) > 0
        assert history[0]["reason"] == reason


class TestSearchPatternManagement:
    """Test search pattern management functionality."""

    @pytest.mark.asyncio
    async def test_search_pattern_lifecycle(self, state_machine):
        """Test complete search pattern lifecycle."""
        from datetime import UTC, datetime

        from src.backend.services.search_pattern_generator import (
            CenterRadiusBoundary,
            PatternType,
            SearchPattern,
        )

        # Set signal processor to satisfy guard condition
        state_machine.set_signal_processor(MagicMock())

        # Create pattern
        pattern = SearchPattern(
            id="test-pattern",
            pattern_type=PatternType.SPIRAL,
            spacing=50.0,
            velocity=5.0,
            boundary=CenterRadiusBoundary(0.0, 0.0, 100.0),
            waypoints=[],
            total_waypoints=10,
            completed_waypoints=0,
            state="IDLE",
            progress_percent=0.0,
            estimated_time_remaining=0.0,
            created_at=datetime.now(UTC),
            started_at=None,
            paused_at=None,
        )

        # Set pattern
        state_machine.set_search_pattern(pattern)
        assert state_machine.get_search_pattern() == pattern

        # Start searching
        await state_machine.transition_to(SystemState.SEARCHING)
        result = await state_machine.start_search_pattern()
        assert result is True
        assert state_machine.get_search_substate() == SearchSubstate.EXECUTING

        # Pause pattern
        result = await state_machine.pause_search_pattern()
        assert result is True
        assert state_machine.get_search_substate() == SearchSubstate.PAUSED

        # Resume pattern
        result = await state_machine.resume_search_pattern()
        assert result is True
        assert state_machine.get_search_substate() == SearchSubstate.EXECUTING

        # Stop pattern
        result = await state_machine.stop_search_pattern()
        assert result is True
        assert state_machine.get_search_substate() == SearchSubstate.IDLE

    @pytest.mark.asyncio
    async def test_waypoint_progress_tracking(self, state_machine):
        """Test waypoint progress tracking."""
        from datetime import UTC, datetime

        from src.backend.services.search_pattern_generator import (
            CenterRadiusBoundary,
            PatternType,
            SearchPattern,
            Waypoint,
        )

        # Create waypoints
        waypoints = [Waypoint(i, 0.0, 0.0, 50.0) for i in range(5)]

        pattern = SearchPattern(
            id="test",
            pattern_type=PatternType.LAWNMOWER,
            spacing=10.0,
            velocity=5.0,
            boundary=CenterRadiusBoundary(0.0, 0.0, 100.0),
            waypoints=waypoints,
            total_waypoints=5,
            completed_waypoints=0,
            state="IDLE",
            progress_percent=0.0,
            estimated_time_remaining=0.0,
            created_at=datetime.now(UTC),
            started_at=None,
            paused_at=None,
        )

        state_machine.set_search_pattern(pattern)

        # Update progress
        state_machine.update_waypoint_progress(3)

        status = state_machine.get_search_pattern_status()
        assert status["completed_waypoints"] == 3
        assert status["progress_percent"] == 60.0

        # Complete pattern
        state_machine.update_waypoint_progress(5)
        assert pattern.state == "COMPLETED"


class TestStatistics:
    """Test state machine statistics."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, state_machine):
        """Test that statistics are properly tracked."""
        # Set signal processor to satisfy guard condition
        state_machine.set_signal_processor(MagicMock())

        # Make some transitions
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.handle_detection(rssi=-55.0, confidence=75.0)
        await state_machine.handle_detection(rssi=-52.0, confidence=80.0)

        stats = state_machine.get_statistics()

        assert stats["current_state"] == "DETECTING"
        assert stats["detection_count"] == 2
        assert stats["state_changes"] == 2
        assert "time_since_detection" in stats

    @pytest.mark.asyncio
    async def test_history_limit(self, state_machine):
        """Test that history limit works correctly."""
        # Set signal processor to satisfy guard condition
        state_machine.set_signal_processor(MagicMock())

        # Make many transitions
        for _i in range(5):
            await state_machine.transition_to(SystemState.SEARCHING)
            await state_machine.transition_to(SystemState.IDLE)

        # Get limited history
        history = state_machine.get_state_history(limit=3)
        assert len(history) == 3

        # Get all history
        all_history = state_machine.get_state_history(limit=0)
        assert len(all_history) == 10


class TestConcurrency:
    """Test concurrent state operations."""

    @pytest.mark.asyncio
    async def test_concurrent_transitions(self, state_machine):
        """Test handling of concurrent transition requests."""
        # Set signal processor to satisfy guard condition
        state_machine.set_signal_processor(MagicMock())

        # Try concurrent transitions
        tasks = [
            state_machine.transition_to(SystemState.SEARCHING),
            state_machine.transition_to(SystemState.SEARCHING),
            state_machine.transition_to(SystemState.SEARCHING),
        ]

        results = await asyncio.gather(*tasks)

        # At least one should succeed
        assert any(results)
        assert state_machine.get_current_state() == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_concurrent_callbacks(self, state_machine):
        """Test that multiple callbacks are executed."""
        callback_times = []

        async def slow_callback(old, new, reason):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.01)
            callback_times.append(asyncio.get_event_loop().time() - start)

        # Set signal processor to satisfy guard condition
        state_machine.set_signal_processor(MagicMock())

        # Add multiple callbacks
        for _ in range(3):
            state_machine.add_state_callback(slow_callback)

        # Transition
        await state_machine.transition_to(SystemState.SEARCHING)

        # Callbacks should be executed (may run sequentially in the test)
        assert len(callback_times) == 3
        assert all(t >= 0.009 for t in callback_times)
