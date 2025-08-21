"""Unit tests for StateMachine debounced transitions per FR7.

Tests debounced state transitions with configurable trigger/drop thresholds
to prevent false positive/negative transitions during signal fluctuations.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.backend.core.exceptions import StateTransitionError
from src.backend.services.state_machine import StateMachine, SystemState


class TestStateMachineDebounce:
    """Test debounced state transitions per FR7 requirements."""

    @pytest.fixture
    def state_machine(self):
        """Provide StateMachine instance with test configuration."""
        return StateMachine(enable_persistence=False)

    @pytest.mark.asyncio
    async def test_configurable_trigger_threshold_default_12db(self, state_machine):
        """Test FR7: Default trigger threshold is 12dB SNR."""
        # TDD RED: This should fail because trigger_threshold_db doesn't exist yet

        # Test that state machine has configurable trigger threshold
        assert hasattr(state_machine, "_trigger_threshold_db")
        assert state_machine._trigger_threshold_db == 12.0

        # Test that threshold is loaded from configuration
        assert state_machine._config_loaded is True

    @pytest.mark.asyncio
    async def test_configurable_drop_threshold_default_6db(self, state_machine):
        """Test FR7: Default drop threshold is 6dB SNR."""
        # TDD RED: This should fail because drop_threshold_db doesn't exist yet

        # Test that state machine has configurable drop threshold
        assert hasattr(state_machine, "_drop_threshold_db")
        assert state_machine._drop_threshold_db == 6.0

        # Test proper separation between thresholds (minimum 3dB)
        separation = state_machine._trigger_threshold_db - state_machine._drop_threshold_db
        assert separation >= 3.0

    @pytest.mark.asyncio
    async def test_threshold_validation_prevents_invalid_config(self, state_machine):
        """Test threshold validation ensures trigger > drop with minimum separation."""
        # TDD RED: This should fail because validation doesn't exist yet

        # Test invalid configuration where trigger <= drop
        with pytest.raises(
            StateTransitionError, match="trigger threshold.*must be greater than drop threshold"
        ):
            await state_machine._load_debounce_config(
                {
                    "trigger_threshold_db": 5.0,
                    "drop_threshold_db": 8.0,  # Invalid: drop > trigger
                }
            )

        # Test invalid configuration with insufficient separation
        with pytest.raises(StateTransitionError, match="minimum 3dB separation required"):
            await state_machine._load_debounce_config(
                {
                    "trigger_threshold_db": 7.0,
                    "drop_threshold_db": 6.0,  # Invalid: only 1dB separation
                }
            )

    @pytest.mark.asyncio
    async def test_signal_processor_integration_with_thresholds(self, state_machine):
        """Test integration with SignalProcessor.process_detection_with_debounce()."""
        # TDD RED: This should fail because integration doesn't exist yet

        # Mock signal processor with real debounce method signature
        mock_signal_processor = Mock()
        mock_signal_processor.process_detection_with_debounce = Mock(return_value=False)
        state_machine._signal_processor = mock_signal_processor

        # Test that state machine uses configured thresholds with signal processor
        rssi = -67.0  # 13dB SNR with -80dBm noise floor
        noise_floor = -80.0

        await state_machine._evaluate_signal_for_transition(rssi, noise_floor)

        # Verify signal processor called with configured thresholds
        mock_signal_processor.process_detection_with_debounce.assert_called_with(
            rssi=rssi,
            noise_floor=noise_floor,
            threshold=12.0,  # trigger_threshold_db
            drop_threshold=6.0,  # drop_threshold_db
        )

    @pytest.mark.asyncio
    async def test_time_based_debounce_periods_configurable(self, state_machine):
        """Test [33a,33b] time-based debouncing with configurable periods."""
        # TDD RED: This should fail because time-based debouncing doesn't exist yet

        # Test that state machine has configurable debounce periods
        assert hasattr(state_machine, "_debounce_detection_period_ms")
        assert hasattr(state_machine, "_debounce_loss_period_ms")

        # Test default values per FR7
        assert state_machine._debounce_detection_period_ms == 300  # 300ms default
        assert state_machine._debounce_loss_period_ms == 300  # 300ms default

    @pytest.mark.asyncio
    async def test_sustained_signal_detection_transition(self, state_machine):
        """Test [33c] transition confirmation requires sustained signal conditions."""
        # TDD RED: This should fail because sustained detection logic doesn't exist yet

        # Setup state machine in SEARCHING state
        await state_machine.transition_to(SystemState.SEARCHING)

        # Mock signal processor for sustained detection
        mock_signal_processor = Mock()
        state_machine._signal_processor = mock_signal_processor

        # First detection should not immediately trigger state change
        mock_signal_processor.process_detection_with_debounce.return_value = True

        # Should require sustained detection over debounce period
        detection_start_time = time.time()
        result = await state_machine._process_sustained_detection(-67.0, -80.0)

        # Should not transition immediately
        assert state_machine.current_state == SystemState.SEARCHING
        assert result is False  # Not sustained yet

        # Simulate sustained detection over time period
        await asyncio.sleep(0.31)  # Slightly over 300ms period
        result = await state_machine._process_sustained_detection(-67.0, -80.0)

        # Should transition after sustained period
        assert result is True  # Sustained detection confirmed

    @pytest.mark.asyncio
    async def test_signal_loss_debounce_prevents_premature_transition(self, state_machine):
        """Test [33d] transition cancellation when conditions change during debounce."""
        # TDD RED: This should fail because signal loss debouncing doesn't exist yet

        # Setup state machine in DETECTING state via valid path: IDLE -> SEARCHING -> DETECTING
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)

        # Mock signal processor for signal loss then recovery
        mock_signal_processor = Mock()
        state_machine._signal_processor = mock_signal_processor

        # Initial signal loss should not immediately drop detection
        mock_signal_processor.process_detection_with_debounce.return_value = False

        loss_start_time = time.time()
        result = await state_machine._process_signal_loss(-85.0, -80.0)

        # Should not transition immediately on signal loss
        assert state_machine.current_state == SystemState.DETECTING
        assert result is False  # Not lost yet

        # Signal recovery before debounce period expires should cancel transition
        mock_signal_processor.process_detection_with_debounce.return_value = True
        await asyncio.sleep(0.15)  # 150ms - halfway through debounce period

        result = await state_machine._process_signal_recovery(-67.0, -80.0)

        # Should cancel the loss transition and remain in DETECTING
        assert state_machine.current_state == SystemState.DETECTING
        assert result is True  # Recovery confirmed, loss cancelled

    @pytest.mark.asyncio
    async def test_comprehensive_transition_logging_with_debounce_details(self, state_machine):
        """Test [33f] comprehensive transition logging with debounce timing and threshold details."""
        # TDD RED: This should fail because enhanced logging doesn't exist yet

        # Setup mock logger to capture log messages
        with patch("src.backend.services.state_machine.logger") as mock_logger:
            await state_machine.transition_to(SystemState.SEARCHING)

            # Trigger debounced transition process
            mock_signal_processor = Mock()
            mock_signal_processor.process_detection_with_debounce.return_value = True
            state_machine._signal_processor = mock_signal_processor

            # Call _process_sustained_detection which should log debounce details
            await state_machine._process_sustained_detection(-67.0, -80.0)

            # Check that info method was called at least once
            assert mock_logger.info.called

            # Get all log messages from info calls
            log_calls = []
            for call in mock_logger.info.call_args_list:
                if call.args:
                    log_calls.append(call.args[0])

            # Verify that at least one log message contains debounce information
            # The _process_sustained_detection method should log: "Started detection debounce: trigger=12.0dB, period=300ms"
            debounce_logs = [msg for msg in log_calls if "debounce" in msg.lower()]
            assert len(debounce_logs) > 0, f"Expected debounce logs, got: {log_calls}"

            # Verify threshold information is logged
            threshold_logs = [
                msg for msg in log_calls if "trigger" in msg.lower() or "threshold" in msg.lower()
            ]
            assert len(threshold_logs) > 0, f"Expected threshold logs, got: {log_calls}"


if __name__ == "__main__":
    pytest.main([__file__])
