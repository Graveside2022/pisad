"""Comprehensive unit tests for state machine covering all state transitions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.backend.services.state_machine import (
    SearchSubstate,
    StateChangeEvent,
    StateMachine,
    SystemState,
)


class TestStateMachineCore:
    """Core state machine functionality tests."""

    @pytest.fixture
    def state_machine(self):
        """Create a state machine with mocked dependencies."""
        sm = StateMachine(enable_persistence=False)

        # Mock all external dependencies
        sm._mavlink_service = MagicMock()
        sm._mavlink_service.send_heartbeat = AsyncMock()
        sm._mavlink_service.send_telemetry = AsyncMock()

        sm._signal_processor = MagicMock()
        sm._signal_processor.is_running = True
        sm._signal_processor.get_current_snr = Mock(return_value=15.0)
        sm._signal_processor.get_current_rssi = Mock(return_value=-75.0)

        return sm

    def test_initialization(self, state_machine):
        """Test state machine initializes correctly."""
        assert state_machine._current_state == SystemState.IDLE
        assert state_machine._previous_state == SystemState.IDLE
        assert state_machine._state_history == []
        assert not state_machine._is_running
        assert not state_machine._homing_enabled

    @pytest.mark.asyncio
    async def test_start_stop(self, state_machine):
        """Test starting and stopping the state machine."""
        # Start
        await state_machine.start()
        assert state_machine._is_running

        # Start again should not fail
        await state_machine.start()
        assert state_machine._is_running

        # Stop
        await state_machine.stop()
        assert not state_machine._is_running

    @pytest.mark.asyncio
    async def test_get_current_state(self, state_machine):
        """Test getting current state."""
        assert state_machine.get_current_state() == SystemState.IDLE

        state_machine._current_state = SystemState.SEARCHING
        assert state_machine.get_current_state() == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_get_status(self, state_machine):
        """Test getting full status."""
        await state_machine.start()
        status = await state_machine.get_status()

        assert status["current_state"] == "IDLE"
        assert status["previous_state"] == "IDLE"
        assert status["is_running"] is True
        assert status["homing_enabled"] is False
        assert "state_entered_time" in status
        assert "time_in_state" in status

    def test_is_homing_allowed(self, state_machine):
        """Test homing permission check."""
        # Not allowed in IDLE
        assert not state_machine.is_homing_allowed()

        # Allowed in DETECTING
        state_machine._current_state = SystemState.DETECTING
        assert state_machine.is_homing_allowed()

        # Allowed in HOLDING
        state_machine._current_state = SystemState.HOLDING
        assert state_machine.is_homing_allowed()

    def test_add_state_callback(self, state_machine):
        """Test adding state change callbacks."""
        callback = Mock()
        state_machine.add_state_callback(callback)
        assert callback in state_machine._state_callbacks

    @pytest.mark.asyncio
    async def test_set_signal_processor(self, state_machine):
        """Test setting signal processor."""
        processor = MagicMock()
        await state_machine.set_signal_processor(processor)
        assert state_machine._signal_processor == processor

    @pytest.mark.asyncio
    async def test_set_mavlink_service(self, state_machine):
        """Test setting MAVLink service."""
        service = MagicMock()
        await state_machine.set_mavlink_service(service)
        assert state_machine._mavlink_service == service


class TestStateTransitions:
    """Test all state transitions."""

    @pytest.fixture
    def state_machine(self):
        """Create a fully mocked state machine."""
        sm = StateMachine(enable_persistence=False)

        # Mock dependencies
        sm._mavlink_service = MagicMock()
        sm._signal_processor = MagicMock()
        sm._signal_processor.is_running = True
        sm._signal_processor.get_current_snr = Mock(return_value=15.0)

        return sm

    @pytest.mark.asyncio
    async def test_idle_to_searching(self, state_machine):
        """Test transition from IDLE to SEARCHING."""
        await state_machine.start()

        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is True
        assert state_machine._current_state == SystemState.SEARCHING
        assert state_machine._previous_state == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_searching_to_detecting(self, state_machine):
        """Test transition from SEARCHING to DETECTING."""
        await state_machine.start()
        state_machine._current_state = SystemState.SEARCHING

        result = await state_machine.transition_to(SystemState.DETECTING)
        assert result is True
        assert state_machine._current_state == SystemState.DETECTING

    @pytest.mark.asyncio
    async def test_detecting_to_homing(self, state_machine):
        """Test transition from DETECTING to HOMING."""
        await state_machine.start()
        state_machine._current_state = SystemState.DETECTING
        state_machine._homing_enabled = True

        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is True
        assert state_machine._current_state == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_homing_to_holding(self, state_machine):
        """Test transition from HOMING to HOLDING."""
        await state_machine.start()
        state_machine._current_state = SystemState.HOMING

        result = await state_machine.transition_to(SystemState.HOLDING)
        assert result is True
        assert state_machine._current_state == SystemState.HOLDING

    @pytest.mark.asyncio
    async def test_holding_to_homing(self, state_machine):
        """Test transition from HOLDING back to HOMING."""
        await state_machine.start()
        state_machine._current_state = SystemState.HOLDING
        state_machine._homing_enabled = True

        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is True
        assert state_machine._current_state == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_any_to_idle(self, state_machine):
        """Test transition from any state to IDLE."""
        await state_machine.start()

        for state in [
            SystemState.SEARCHING,
            SystemState.DETECTING,
            SystemState.HOMING,
            SystemState.HOLDING,
        ]:
            state_machine._current_state = state
            result = await state_machine.transition_to(SystemState.IDLE)
            assert result is True
            assert state_machine._current_state == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_invalid_transition(self, state_machine):
        """Test invalid state transitions."""
        await state_machine.start()

        # IDLE to HOMING is invalid
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is False
        assert state_machine._current_state == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_force_transition(self, state_machine):
        """Test force transition bypasses validation."""
        await state_machine.start()

        # Force IDLE to HOMING (normally invalid)
        result = await state_machine.force_transition(SystemState.HOMING)
        assert result is True
        assert state_machine._current_state == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_transition_with_reason(self, state_machine):
        """Test state transition with reason."""
        await state_machine.start()

        reason = "Signal detected"
        state_machine._current_state = SystemState.SEARCHING
        await state_machine.transition_to(SystemState.DETECTING, reason=reason)

        # Check history
        assert len(state_machine._state_history) > 0
        last_event = state_machine._state_history[-1]
        assert last_event.reason == reason


class TestStateCallbacks:
    """Test state change callbacks."""

    @pytest.fixture
    def state_machine(self):
        """Create mocked state machine."""
        sm = StateMachine(enable_persistence=False)
        sm._mavlink_service = MagicMock()
        sm._signal_processor = MagicMock()
        sm._signal_processor.is_running = True
        return sm

    @pytest.mark.asyncio
    async def test_state_callbacks_called(self, state_machine):
        """Test callbacks are called on state change."""
        callback = Mock()
        state_machine.add_state_callback(callback)

        await state_machine.start()
        await state_machine.transition_to(SystemState.SEARCHING)

        callback.assert_called()
        call_args = callback.call_args[0][0]
        assert isinstance(call_args, StateChangeEvent)
        assert call_args.from_state == SystemState.IDLE
        assert call_args.to_state == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_callback_exception_handled(self, state_machine):
        """Test callback exceptions don't break transitions."""

        def bad_callback(event):
            raise Exception("Callback error")

        state_machine.add_state_callback(bad_callback)

        await state_machine.start()
        # Should not raise exception
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is True


class TestSearchPatternManagement:
    """Test search pattern state management."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine with search pattern support."""
        sm = StateMachine(enable_persistence=False)
        sm._mavlink_service = MagicMock()
        sm._signal_processor = MagicMock()
        sm._signal_processor.is_running = True
        return sm

    @pytest.mark.asyncio
    async def test_start_search_pattern(self, state_machine):
        """Test starting a search pattern."""
        pattern = MagicMock()
        pattern.waypoints = [MagicMock() for _ in range(5)]

        await state_machine.start()
        await state_machine.start_search_pattern(pattern)

        assert state_machine._active_pattern == pattern
        assert state_machine._search_substate == SearchSubstate.EXECUTING
        assert state_machine._current_waypoint_index == 0

    @pytest.mark.asyncio
    async def test_pause_search_pattern(self, state_machine):
        """Test pausing a search pattern."""
        pattern = MagicMock()
        pattern.waypoints = [MagicMock() for _ in range(5)]

        await state_machine.start()
        await state_machine.start_search_pattern(pattern)
        await state_machine.pause_search_pattern()

        assert state_machine._search_substate == SearchSubstate.PAUSED
        assert state_machine._pattern_paused_at is not None

    @pytest.mark.asyncio
    async def test_resume_search_pattern(self, state_machine):
        """Test resuming a paused search pattern."""
        pattern = MagicMock()
        pattern.waypoints = [MagicMock() for _ in range(5)]

        await state_machine.start()
        await state_machine.start_search_pattern(pattern)
        await state_machine.pause_search_pattern()
        await state_machine.resume_search_pattern()

        assert state_machine._search_substate == SearchSubstate.EXECUTING
        assert state_machine._pattern_paused_at is None

    @pytest.mark.asyncio
    async def test_stop_search_pattern(self, state_machine):
        """Test stopping a search pattern."""
        pattern = MagicMock()
        pattern.waypoints = [MagicMock() for _ in range(5)]

        await state_machine.start()
        await state_machine.start_search_pattern(pattern)
        await state_machine.stop_search_pattern()

        assert state_machine._search_substate == SearchSubstate.IDLE
        assert state_machine._active_pattern is None
        assert state_machine._current_waypoint_index == 0

    @pytest.mark.asyncio
    async def test_get_search_status(self, state_machine):
        """Test getting search pattern status."""
        pattern = MagicMock()
        pattern.waypoints = [MagicMock() for _ in range(5)]
        pattern.pattern_type = "SPIRAL"

        await state_machine.start()
        await state_machine.start_search_pattern(pattern)

        status = state_machine.get_search_status()
        assert status["substate"] == "EXECUTING"
        assert status["pattern_type"] == "SPIRAL"
        assert status["current_waypoint"] == 0
        assert status["total_waypoints"] == 5
        assert status["is_paused"] is False


class TestStateTimeouts:
    """Test state timeout functionality."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine with fast timeouts."""
        sm = StateMachine(enable_persistence=False)
        sm._mavlink_service = MagicMock()
        sm._signal_processor = MagicMock()
        sm._signal_processor.is_running = True

        # Set very short timeouts for testing
        sm._state_timeouts = {
            SystemState.IDLE: 0,
            SystemState.SEARCHING: 0.1,
            SystemState.DETECTING: 0.1,
            SystemState.HOMING: 0.1,
            SystemState.HOLDING: 0.1,
        }
        return sm

    @pytest.mark.asyncio
    async def test_state_timeout_triggers(self, state_machine):
        """Test that state timeout triggers transition to IDLE."""
        await state_machine.start()
        await state_machine.transition_to(SystemState.SEARCHING)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Should have transitioned to IDLE
        assert state_machine._current_state == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_cancel_timeout_on_transition(self, state_machine):
        """Test timeout is cancelled on state transition."""
        await state_machine.start()
        await state_machine.transition_to(SystemState.SEARCHING)

        # Transition before timeout
        await asyncio.sleep(0.05)
        await state_machine.transition_to(SystemState.DETECTING)

        # Wait past original timeout
        await asyncio.sleep(0.1)

        # Should still be in DETECTING
        assert state_machine._current_state == SystemState.DETECTING


class TestHomingControl:
    """Test homing enable/disable functionality."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine."""
        sm = StateMachine(enable_persistence=False)
        sm._mavlink_service = MagicMock()
        sm._signal_processor = MagicMock()
        sm._signal_processor.is_running = True
        return sm

    @pytest.mark.asyncio
    async def test_enable_homing(self, state_machine):
        """Test enabling homing."""
        await state_machine.start()
        state_machine._current_state = SystemState.DETECTING

        result = await state_machine.enable_homing()
        assert result is True
        assert state_machine._homing_enabled is True

    @pytest.mark.asyncio
    async def test_disable_homing(self, state_machine):
        """Test disabling homing."""
        await state_machine.start()
        state_machine._current_state = SystemState.HOMING
        state_machine._homing_enabled = True

        result = await state_machine.disable_homing()
        assert result is True
        assert state_machine._homing_enabled is False
        assert state_machine._current_state == SystemState.DETECTING

    @pytest.mark.asyncio
    async def test_cannot_enable_homing_in_idle(self, state_machine):
        """Test homing cannot be enabled in IDLE state."""
        await state_machine.start()

        result = await state_machine.enable_homing()
        assert result is False
        assert state_machine._homing_enabled is False


class TestDetectionHandling:
    """Test detection event handling."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine."""
        sm = StateMachine(enable_persistence=False)
        sm._mavlink_service = MagicMock()
        sm._signal_processor = MagicMock()
        sm._signal_processor.is_running = True
        sm._signal_processor.get_current_snr = Mock(return_value=15.0)
        return sm

    @pytest.mark.asyncio
    async def test_handle_detection_from_searching(self, state_machine):
        """Test handling detection while searching."""
        await state_machine.start()
        state_machine._current_state = SystemState.SEARCHING

        await state_machine.handle_detection(rssi=-75.0, snr=15.0, confidence=80.0)

        assert state_machine._current_state == SystemState.DETECTING
        assert state_machine._detection_count > 0

    @pytest.mark.asyncio
    async def test_handle_loss_of_signal(self, state_machine):
        """Test handling loss of signal."""
        await state_machine.start()
        state_machine._current_state = SystemState.DETECTING

        await state_machine.handle_loss_of_signal()

        assert state_machine._current_state == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_auto_transition_to_homing(self, state_machine):
        """Test auto-transition to homing when enabled."""
        await state_machine.start()
        state_machine._current_state = SystemState.DETECTING
        state_machine._homing_enabled = True

        # Mock high SNR
        state_machine._signal_processor.get_current_snr = Mock(return_value=20.0)

        # Trigger check for auto-transition
        await state_machine._check_auto_homing()

        assert state_machine._current_state == SystemState.HOMING


class TestTelemetryIntegration:
    """Test telemetry and MAVLink integration."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine with mocked MAVLink."""
        sm = StateMachine(enable_persistence=False)
        sm._mavlink_service = MagicMock()
        sm._mavlink_service.send_heartbeat = AsyncMock()
        sm._mavlink_service.send_telemetry = AsyncMock()
        sm._signal_processor = MagicMock()
        return sm

    @pytest.mark.asyncio
    async def test_telemetry_sent_periodically(self, state_machine):
        """Test telemetry is sent periodically."""
        await state_machine.start()

        # Start telemetry
        await state_machine._start_telemetry()

        # Wait for telemetry to be sent
        await asyncio.sleep(0.1)

        # Check heartbeat was sent
        state_machine._mavlink_service.send_heartbeat.assert_called()

        # Stop telemetry
        await state_machine._stop_telemetry()

    @pytest.mark.asyncio
    async def test_mavlink_state_update(self, state_machine):
        """Test MAVLink is updated on state change."""
        await state_machine.start()

        # Transition state
        await state_machine.transition_to(SystemState.SEARCHING)

        # Check MAVLink was updated
        state_machine._mavlink_service.send_telemetry.assert_called()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine."""
        sm = StateMachine(enable_persistence=False)
        return sm

    @pytest.mark.asyncio
    async def test_transition_without_dependencies(self, state_machine):
        """Test transitions fail gracefully without dependencies."""
        await state_machine.start()

        # Try to transition to SEARCHING without signal processor
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is False
        assert state_machine._current_state == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_concurrent_transitions(self, state_machine):
        """Test handling concurrent transition requests."""
        state_machine._mavlink_service = MagicMock()
        state_machine._signal_processor = MagicMock()
        state_machine._signal_processor.is_running = True

        await state_machine.start()

        # Start multiple concurrent transitions
        tasks = [
            state_machine.transition_to(SystemState.SEARCHING),
            state_machine.transition_to(SystemState.SEARCHING),
            state_machine.transition_to(SystemState.SEARCHING),
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed or fail gracefully
        assert all(isinstance(r, bool) for r in results)
        assert state_machine._current_state in SystemState

    @pytest.mark.asyncio
    async def test_stop_during_transition(self, state_machine):
        """Test stopping during a state transition."""
        state_machine._mavlink_service = MagicMock()
        state_machine._signal_processor = MagicMock()
        state_machine._signal_processor.is_running = True

        await state_machine.start()

        # Start transition and stop concurrently
        transition_task = asyncio.create_task(state_machine.transition_to(SystemState.SEARCHING))
        stop_task = asyncio.create_task(state_machine.stop())

        await asyncio.gather(transition_task, stop_task)

        assert not state_machine._is_running
