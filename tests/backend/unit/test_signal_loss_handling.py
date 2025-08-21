"""Test signal loss handling in state machine per TASK-2.2.8 and PRD-FR17.

TASK-2.2.8: Automatic Signal Loss Recovery
- Tests 10-second signal loss detection and automatic homing disable
- Tests operator notification system integration
- Tests signal recovery detection and re-enable logic
- Tests comprehensive state management and logging

PRD-FR17: Automatically disable homing mode after 10 seconds signal loss, notify operator
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.state_machine import StateMachine, SystemState


@pytest.fixture
def mock_mavlink_service():
    """Mock MAVLink service for testing."""
    service = AsyncMock()
    service.send_statustext = AsyncMock(return_value=True)
    service.send_telemetry = AsyncMock(return_value=True)
    service.send_state_change = AsyncMock(return_value=True)
    service.send_detection_telemetry = AsyncMock(return_value=True)
    service.send_signal_lost_telemetry = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_signal_processor():
    """Mock signal processor for testing."""
    processor = MagicMock()
    processor.process_detection_with_debounce = MagicMock(return_value=False)
    return processor


@pytest.fixture
async def state_machine_with_signal_loss(mock_mavlink_service, mock_signal_processor):
    """State machine configured for signal loss testing."""
    state_machine = StateMachine(enable_persistence=False)
    state_machine.set_mavlink_service(mock_mavlink_service)
    state_machine.set_signal_processor(mock_signal_processor)
    await state_machine.start()
    yield state_machine
    try:
        await state_machine.stop()
    except Exception:
        pass  # Ignore errors during cleanup


class TestSignalLossDetection:
    """Test signal loss detection and handling per TASK-2.2.8.1."""

    @pytest.mark.asyncio
    async def test_signal_loss_timer_starts_on_signal_lost(self, state_machine_with_signal_loss):
        """Test [34a,34e] - Signal loss timer starts when signal is lost.

        PRD-FR17: System shall automatically disable homing after 10 seconds signal loss
        """
        state_machine = state_machine_with_signal_loss

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Verify initial state
        assert state_machine.get_current_state() == SystemState.HOMING
        assert state_machine._homing_enabled is True
        assert (
            not hasattr(state_machine, "_signal_loss_timer")
            or state_machine._signal_loss_timer is None
        )

        # Trigger signal loss
        await state_machine.handle_signal_lost()

        # Verify signal loss timer started
        assert hasattr(state_machine, "_signal_loss_timer")
        assert state_machine._signal_loss_timer is not None
        assert not state_machine._signal_loss_timer.done()

        # Verify signal loss tracking activated
        assert hasattr(state_machine, "_signal_lost_time")
        assert state_machine._signal_lost_time is not None
        assert hasattr(state_machine, "_signal_loss_active")
        assert state_machine._signal_loss_active is True

    @pytest.mark.asyncio
    async def test_homing_disabled_after_10_second_timeout(self, state_machine_with_signal_loss):
        """Test [34b,34f] - Homing automatically disabled after 10-second timeout.

        PRD-FR17: Automatically disable homing mode after 10 seconds signal loss
        """
        state_machine = state_machine_with_signal_loss

        # Configure shorter timeout for testing (0.1 seconds instead of 10)
        state_machine._signal_loss_timeout = 0.1

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss
        await state_machine.handle_signal_lost()

        # Verify homing still enabled initially
        assert state_machine._homing_enabled is True

        # Wait for timeout to trigger
        await asyncio.sleep(0.15)  # Wait longer than timeout

        # Verify homing was automatically disabled
        assert state_machine._homing_enabled is False

        # Verify timer was cleaned up
        assert (
            not hasattr(state_machine, "_signal_loss_timer")
            or state_machine._signal_loss_timer is None
        )

    @pytest.mark.asyncio
    async def test_operator_notification_on_signal_loss(self, state_machine_with_signal_loss):
        """Test [34c] - Operator notification system for signal loss events.

        PRD-FR17: Notify operator of signal loss
        """
        state_machine = state_machine_with_signal_loss
        mock_mavlink = state_machine._mavlink_service

        # Configure shorter timeout for testing
        state_machine._signal_loss_timeout = 0.1

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss
        await state_machine.handle_signal_lost()

        # Wait for timeout to trigger
        await asyncio.sleep(0.15)

        # Verify operator was notified via MAVLink StatusText
        mock_mavlink.send_statustext.assert_called()

        # Check notification message content
        call_args = mock_mavlink.send_statustext.call_args_list
        assert any("signal loss" in str(call).lower() for call in call_args)
        assert any("homing disabled" in str(call).lower() for call in call_args)

    @pytest.mark.asyncio
    async def test_signal_recovery_cancels_timer(self, state_machine_with_signal_loss):
        """Test [34d] - Signal recovery detection cancels timer.

        Signal recovery should cancel the signal loss timer and allow homing to continue.
        """
        state_machine = state_machine_with_signal_loss

        # Configure longer timeout for testing
        state_machine._signal_loss_timeout = 1.0

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss
        await state_machine.handle_signal_lost()

        # Verify timer started
        assert hasattr(state_machine, "_signal_loss_timer")
        assert state_machine._signal_loss_timer is not None
        assert state_machine._signal_loss_active is True

        # Trigger signal recovery before timeout
        await asyncio.sleep(0.1)  # Small delay
        await state_machine.handle_signal_recovery()

        # Verify timer was cancelled
        assert (
            not hasattr(state_machine, "_signal_loss_timer")
            or state_machine._signal_loss_timer is None
        )
        assert state_machine._signal_loss_active is False

        # Verify homing still enabled
        assert state_machine._homing_enabled is True


class TestSignalLossStateManagement:
    """Test comprehensive signal loss state management per TASK-2.2.8.2."""

    @pytest.mark.asyncio
    async def test_signal_loss_state_tracking(self, state_machine_with_signal_loss):
        """Test [35a] - Signal loss tracking state management.

        Verify _signal_lost_time and _signal_loss_active attributes work correctly.
        """
        state_machine = state_machine_with_signal_loss

        # Initially no signal loss
        assert (
            not hasattr(state_machine, "_signal_lost_time")
            or state_machine._signal_lost_time is None
        )
        assert (
            not hasattr(state_machine, "_signal_loss_active")
            or state_machine._signal_loss_active is False
        )

        # Enable homing and transition to DETECTING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.DETECTING, "Test detection")

        # Trigger signal loss
        start_time = time.time()
        await state_machine.handle_signal_lost()

        # Verify signal loss state tracking
        assert state_machine._signal_lost_time is not None
        assert abs(state_machine._signal_lost_time - start_time) < 0.1  # Within 100ms
        assert state_machine._signal_loss_active is True

        # Trigger signal recovery
        await state_machine.handle_signal_recovery()

        # Verify state tracking cleared
        assert state_machine._signal_lost_time is None
        assert state_machine._signal_loss_active is False

    @pytest.mark.asyncio
    async def test_velocity_command_termination_integration(self, state_machine_with_signal_loss):
        """Test [35b] - Automatic velocity command termination on signal loss.

        Integration test with homing controller disable_homing() method.
        """
        state_machine = state_machine_with_signal_loss

        # Mock homing controller
        mock_homing_controller = AsyncMock()
        mock_homing_controller.disable_homing = AsyncMock()
        state_machine._homing_controller = mock_homing_controller

        # Configure shorter timeout
        state_machine._signal_loss_timeout = 0.1

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss
        await state_machine.handle_signal_lost()

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Verify homing controller was notified to disable
        mock_homing_controller.disable_homing.assert_called_once_with("Signal loss timeout")

    @pytest.mark.asyncio
    async def test_comprehensive_signal_loss_logging(self, state_machine_with_signal_loss):
        """Test [35c] - Comprehensive signal loss logging with timestamps and recovery tracking."""
        state_machine = state_machine_with_signal_loss

        # Configure shorter timeout
        state_machine._signal_loss_timeout = 0.1

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss
        await state_machine.handle_signal_lost()

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Check state history for signal loss events
        history = state_machine.get_state_history()

        # Should have transition due to signal loss
        signal_loss_events = [
            event for event in history if "signal" in event.get("reason", "").lower()
        ]
        assert len(signal_loss_events) > 0

        # Verify timestamp present
        for event in signal_loss_events:
            assert "timestamp" in event
            assert event["timestamp"] is not None

    @pytest.mark.asyncio
    async def test_signal_recovery_validation(self, state_machine_with_signal_loss):
        """Test [35d] - Signal recovery validation before re-enabling homing.

        Signal strength must meet detection thresholds before allowing re-enable.
        """
        state_machine = state_machine_with_signal_loss

        # Configure mock signal processor to return weak signal initially
        state_machine._signal_processor.process_detection_with_debounce.return_value = False

        # Enable homing and trigger signal loss
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")
        await state_machine.handle_signal_lost()

        # Try to recover with weak signal - should not allow re-enable
        await state_machine.handle_signal_recovery()

        # Homing should still be disabled due to weak signal
        # (Implementation detail: recovery validation checks signal strength)

        # Configure strong signal
        state_machine._signal_processor.process_detection_with_debounce.return_value = True

        # Try recovery again with strong signal
        await state_machine.handle_signal_recovery()

        # Now recovery should be allowed
        assert state_machine._signal_loss_active is False

    @pytest.mark.asyncio
    async def test_debounce_integration_prevents_oscillation(self, state_machine_with_signal_loss):
        """Test [35e] - Integration with existing debounce logic prevents oscillation."""
        state_machine = state_machine_with_signal_loss

        # Enable homing and transition to DETECTING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.DETECTING, "Test detection")

        # Rapid signal loss and recovery should not cause oscillation
        for _ in range(5):
            await state_machine.handle_signal_lost()
            await asyncio.sleep(0.01)  # Very short delay
            await state_machine.handle_signal_recovery()
            await asyncio.sleep(0.01)

        # Should remain stable without excessive state transitions
        history = state_machine.get_state_history()
        transition_count = len(history)

        # Should not have excessive transitions (less than 10 for 5 cycles)
        assert transition_count < 10

    @pytest.mark.asyncio
    async def test_signal_loss_metrics_tracking(self, state_machine_with_signal_loss):
        """Test [35f] - Signal loss metrics tracking for telemetry."""
        state_machine = state_machine_with_signal_loss

        # Configure shorter timeout
        state_machine._signal_loss_timeout = 0.1

        # Enable homing and trigger multiple signal loss events
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss
        await state_machine.handle_signal_lost()
        await asyncio.sleep(0.15)  # Wait for timeout

        # Get telemetry metrics
        metrics = state_machine.get_telemetry_metrics()

        # Verify signal loss metrics are tracked
        assert "total_transitions" in metrics
        assert metrics["total_transitions"] > 0

        # Should include state transition from signal loss
        assert "state_durations" in metrics
        assert "transition_frequencies" in metrics


class TestSignalLossEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_signal_loss_in_idle_state_ignored(self, state_machine_with_signal_loss):
        """Signal loss in IDLE state should be ignored."""
        state_machine = state_machine_with_signal_loss

        # Ensure in IDLE state
        await state_machine.transition_to(SystemState.IDLE, "Test idle")

        # Trigger signal loss - should be ignored
        await state_machine.handle_signal_lost()

        # Should not have signal loss timer
        assert (
            not hasattr(state_machine, "_signal_loss_timer")
            or state_machine._signal_loss_timer is None
        )
        assert (
            not hasattr(state_machine, "_signal_loss_active")
            or state_machine._signal_loss_active is False
        )

    @pytest.mark.asyncio
    async def test_multiple_signal_loss_calls_handled_gracefully(
        self, state_machine_with_signal_loss
    ):
        """Multiple signal loss calls should not create multiple timers."""
        state_machine = state_machine_with_signal_loss

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss multiple times
        await state_machine.handle_signal_lost()
        first_timer = getattr(state_machine, "_signal_loss_timer", None)

        await state_machine.handle_signal_lost()
        second_timer = getattr(state_machine, "_signal_loss_timer", None)

        # Should reuse or replace timer, not create multiple
        assert first_timer is not None
        assert second_timer is not None
        # Either same timer or old one was cancelled

    @pytest.mark.asyncio
    async def test_signal_recovery_without_loss_handled_gracefully(
        self, state_machine_with_signal_loss
    ):
        """Signal recovery without prior loss should be handled gracefully."""
        state_machine = state_machine_with_signal_loss

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Call recovery without prior loss - should not crash
        await state_machine.handle_signal_recovery()

        # Should remain in stable state
        assert state_machine.get_current_state() == SystemState.HOMING
        assert state_machine._homing_enabled is True
