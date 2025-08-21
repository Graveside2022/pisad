"""Comprehensive test for signal loss handling per TASK-2.2.8."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.state_machine import StateMachine, SystemState


@pytest.mark.asyncio
async def test_comprehensive_signal_loss_with_metrics():
    """Test signal loss with comprehensive metrics tracking per TASK-2.2.8 [35f]."""
    state_machine = StateMachine(enable_persistence=False)

    # Mock services
    mock_mavlink = AsyncMock()
    mock_mavlink.send_statustext = AsyncMock(return_value=True)
    mock_signal_processor = MagicMock()
    mock_signal_processor.process_detection_with_debounce = MagicMock(return_value=True)

    state_machine.set_mavlink_service(mock_mavlink)
    state_machine.set_signal_processor(mock_signal_processor)

    # Set short timeout for testing
    state_machine._signal_loss_timeout = 0.1

    try:
        await state_machine.start()

        # Enable homing
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Get initial metrics
        initial_metrics = state_machine.get_telemetry_metrics()
        initial_signal_loss_events = initial_metrics["signal_loss_events"]
        initial_recovery_events = initial_metrics["signal_recovery_events"]

        # Trigger signal loss and immediate recovery
        await state_machine.handle_signal_lost()
        await asyncio.sleep(0.05)  # Short delay
        await state_machine.handle_signal_recovery()

        # Check metrics after recovery
        metrics_after_recovery = state_machine.get_telemetry_metrics()

        assert metrics_after_recovery["signal_loss_events"] == initial_signal_loss_events + 1
        assert metrics_after_recovery["signal_recovery_events"] == initial_recovery_events + 1
        assert len(metrics_after_recovery["signal_loss_durations"]) == 1
        assert metrics_after_recovery["signal_loss_durations"][0] > 0
        assert metrics_after_recovery["average_signal_loss_duration"] > 0

        # Test homing disable metrics
        await state_machine.handle_signal_lost()
        await asyncio.sleep(0.15)  # Wait for timeout

        metrics_after_timeout = state_machine.get_telemetry_metrics()
        assert metrics_after_timeout["homing_disabled_by_signal_loss"] == 1
        assert state_machine._homing_enabled is False

        print("✅ Comprehensive signal loss metrics test passed!")

    finally:
        try:
            await state_machine.stop()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_homing_controller_integration():
    """Test signal loss integration with homing controller per TASK-2.2.8 [35b]."""
    state_machine = StateMachine(enable_persistence=False)

    # Mock services
    mock_mavlink = AsyncMock()
    mock_mavlink.send_statustext = AsyncMock(return_value=True)
    mock_signal_processor = MagicMock()
    mock_homing_controller = AsyncMock()
    mock_homing_controller.disable_homing = AsyncMock()

    state_machine.set_mavlink_service(mock_mavlink)
    state_machine.set_signal_processor(mock_signal_processor)
    state_machine.set_homing_controller(mock_homing_controller)

    # Set short timeout
    state_machine._signal_loss_timeout = 0.1

    try:
        await state_machine.start()

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.HOMING, "Test homing")

        # Trigger signal loss
        await state_machine.handle_signal_lost()

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Verify homing controller was notified to disable
        mock_homing_controller.disable_homing.assert_called_once_with("Signal loss timeout")

        # Verify homing was disabled in state machine
        assert state_machine._homing_enabled is False

        print("✅ Homing controller integration test passed!")

    finally:
        try:
            await state_machine.stop()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_signal_loss_edge_cases():
    """Test signal loss edge cases and error handling."""
    state_machine = StateMachine(enable_persistence=False)

    # Mock services
    mock_mavlink = AsyncMock()
    mock_mavlink.send_statustext = AsyncMock(return_value=True)
    mock_signal_processor = MagicMock()

    state_machine.set_mavlink_service(mock_mavlink)
    state_machine.set_signal_processor(mock_signal_processor)

    try:
        await state_machine.start()

        # Test signal loss in IDLE state - should be ignored
        await state_machine.transition_to(SystemState.IDLE, "Test idle")
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

        # Test signal recovery without prior loss - should not crash
        await state_machine.transition_to(SystemState.HOMING, "Test homing")
        state_machine.enable_homing(True)
        await state_machine.handle_signal_recovery()

        # Should remain in stable state
        assert state_machine.get_current_state() == SystemState.HOMING
        assert state_machine._homing_enabled is True

        # Test multiple signal loss calls - should handle gracefully
        await state_machine.handle_signal_lost()
        first_timer = getattr(state_machine, "_signal_loss_timer", None)

        await state_machine.handle_signal_lost()
        second_timer = getattr(state_machine, "_signal_loss_timer", None)

        # Should handle multiple calls gracefully
        assert first_timer is not None
        assert second_timer is not None

        print("✅ Signal loss edge cases test passed!")

    finally:
        try:
            await state_machine.stop()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_signal_loss_debounce_integration():
    """Test signal loss integration with existing debounce logic per TASK-2.2.8 [35e]."""
    state_machine = StateMachine(enable_persistence=False)

    # Mock services
    mock_mavlink = AsyncMock()
    mock_mavlink.send_statustext = AsyncMock(return_value=True)
    mock_signal_processor = MagicMock()

    state_machine.set_mavlink_service(mock_mavlink)
    state_machine.set_signal_processor(mock_signal_processor)

    try:
        await state_machine.start()

        # Enable homing and transition to DETECTING state
        state_machine.enable_homing(True)
        await state_machine.transition_to(SystemState.DETECTING, "Test detection")

        # Rapid signal loss and recovery cycles should not cause excessive transitions
        for _ in range(5):
            await state_machine.handle_signal_lost()
            await asyncio.sleep(0.01)  # Very short delay
            await state_machine.handle_signal_recovery()
            await asyncio.sleep(0.01)

        # Should remain stable without excessive state transitions
        history = state_machine.get_state_history()
        transition_count = len(history)

        # Should not have excessive transitions (reasonable limit for 5 cycles)
        assert transition_count < 15  # Allow some transitions but prevent oscillation

        print("✅ Signal loss debounce integration test passed!")

    finally:
        try:
            await state_machine.stop()
        except Exception:
            pass


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_comprehensive_signal_loss_with_metrics())
    asyncio.run(test_homing_controller_integration())
    asyncio.run(test_signal_loss_edge_cases())
    asyncio.run(test_signal_loss_debounce_integration())
