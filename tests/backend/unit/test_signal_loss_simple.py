"""Simple test for signal loss handling to verify functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.state_machine import StateMachine, SystemState


@pytest.mark.asyncio
async def test_signal_loss_basic_functionality():
    """Test basic signal loss functionality without complex setup."""
    # Create state machine with minimal configuration
    state_machine = StateMachine(enable_persistence=False)

    # Mock services
    mock_mavlink = AsyncMock()
    mock_mavlink.send_statustext = AsyncMock(return_value=True)
    mock_signal_processor = MagicMock()
    mock_signal_processor.process_detection_with_debounce = MagicMock(return_value=False)

    state_machine.set_mavlink_service(mock_mavlink)
    state_machine.set_signal_processor(mock_signal_processor)

    # Set shorter timeout for testing
    state_machine._signal_loss_timeout = 0.1

    try:
        # Start the state machine
        await state_machine.start()

        # Enable homing and transition to HOMING state
        state_machine.enable_homing(True)
        success = await state_machine.transition_to(SystemState.HOMING, "Test homing")
        assert success
        assert state_machine.get_current_state() == SystemState.HOMING

        # Verify initial state
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

        # Wait for timeout to trigger
        await asyncio.sleep(0.15)  # Wait longer than timeout

        # Verify homing was automatically disabled
        assert state_machine._homing_enabled is False

        # Verify operator was notified via MAVLink StatusText
        mock_mavlink.send_statustext.assert_called()

        print("✅ Basic signal loss functionality test passed!")

    finally:
        # Clean up
        try:
            await state_machine.stop()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_signal_recovery_functionality():
    """Test signal recovery cancels timer."""
    state_machine = StateMachine(enable_persistence=False)

    # Mock services
    mock_mavlink = AsyncMock()
    mock_mavlink.send_statustext = AsyncMock(return_value=True)
    mock_signal_processor = MagicMock()
    mock_signal_processor.process_detection_with_debounce = MagicMock(return_value=True)

    state_machine.set_mavlink_service(mock_mavlink)
    state_machine.set_signal_processor(mock_signal_processor)

    # Set longer timeout for testing
    state_machine._signal_loss_timeout = 1.0

    try:
        await state_machine.start()

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

        print("✅ Signal recovery functionality test passed!")

    finally:
        try:
            await state_machine.stop()
        except Exception:
            pass


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_signal_loss_basic_functionality())
    asyncio.run(test_signal_recovery_functionality())
