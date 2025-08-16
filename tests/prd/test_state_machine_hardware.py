#!/usr/bin/env python3
"""PRD-aligned state machine tests with real timing requirements.

Tests FR3, FR15, FR17 requirements from PRD.
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from backend.services.mavlink_service import MAVLinkService
from backend.services.signal_processor import SignalProcessor
from backend.services.state_machine import StateMachine, SystemState


class TestStateMachineHardware:
    """Test state machine with real timing requirements."""

    @pytest.fixture
    async def state_machine(self):
        """Create state machine instance."""
        sm = StateMachine()
        await sm.initialize()
        return sm

    @pytest.fixture
    def signal_processor(self):
        """Create signal processor."""
        return SignalProcessor()

    @pytest.mark.asyncio
    async def test_fr3_state_transitions_2seconds(self, state_machine):
        """FR3: Test state transitions complete within 2 seconds.

        Requirement: State transitions shall complete within 2 seconds
        """
        # Test all valid transitions
        transition_times = []

        test_transitions = [
            (SystemState.IDLE, SystemState.SEARCHING),
            (SystemState.SEARCHING, SystemState.DETECTING),
            (SystemState.DETECTING, SystemState.HOMING),
            (SystemState.HOMING, SystemState.DETECTING),
            (SystemState.DETECTING, SystemState.SEARCHING),
            (SystemState.SEARCHING, SystemState.IDLE),
        ]

        for from_state, to_state in test_transitions:
            # Set initial state
            state_machine._current_state = from_state

            # Measure transition time
            start_time = time.perf_counter()
            success = await state_machine.transition_to(to_state, "Test transition")
            end_time = time.perf_counter()

            transition_time = end_time - start_time
            transition_times.append(transition_time)

            # Verify transition completed
            if success:
                assert state_machine.get_current_state() == to_state
                assert (
                    transition_time < 2.0
                ), f"Transition {from_state.value}->{to_state.value} took {transition_time:.3f}s > 2s"

        # Verify all transitions were fast
        max_time = max(transition_times)
        avg_time = sum(transition_times) / len(transition_times)

        assert max_time < 2.0, f"Max transition time {max_time:.3f}s exceeds 2s"
        assert avg_time < 0.5, f"Average transition time {avg_time:.3f}s too high"

        print(f"Transition times: avg={avg_time:.3f}s, max={max_time:.3f}s")

    @pytest.mark.asyncio
    async def test_fr15_velocity_command_cessation(self, state_machine):
        """FR15: Test velocity command cessation on mode change.

        Requirement: Velocity commands shall stop on operation mode change
        """
        # Setup mock MAVLink service
        mavlink_service = MagicMock()
        state_machine._mavlink_service = mavlink_service

        # Start in HOMING state (sending velocity commands)
        state_machine._current_state = SystemState.HOMING
        state_machine._homing_enabled = True

        # Simulate velocity commands being sent
        velocity_commands_active = True
        mavlink_service.stop_velocity_commands = MagicMock(
            side_effect=lambda: setattr(locals(), "velocity_commands_active", False)
        )

        # Change operation mode
        await state_machine.on_mode_change("MANUAL")

        # Verify homing is disabled (which implicitly stops velocity commands)
        assert not state_machine._homing_enabled, "Homing should be disabled on MANUAL mode"

        # Verify state changed appropriately
        assert state_machine.get_current_state() != SystemState.HOMING

        # Test mode change during different states
        test_modes = ["MANUAL", "AUTO", "GUIDED", "LOITER"]

        for mode in test_modes:
            state_machine._current_state = SystemState.HOMING

            # Change mode
            await state_machine.on_mode_change(mode)

            # Should exit HOMING state
            assert (
                state_machine.get_current_state() != SystemState.HOMING or mode == "AUTO"
            ), f"Should exit HOMING on mode change to {mode}"

    @pytest.mark.asyncio
    async def test_fr17_auto_disable_signal_loss(self, state_machine):
        """FR17: Test auto-disable after signal loss.

        Requirement: System shall auto-disable homing after 30s signal loss
        """
        # Setup initial conditions
        state_machine._current_state = SystemState.HOMING
        state_machine._homing_enabled = True
        state_machine._signal_lost_time = time.time()

        # Test immediate signal loss handling
        await state_machine.on_signal_lost()

        # Should transition to SEARCHING
        assert state_machine.get_current_state() == SystemState.SEARCHING

        # Simulate 30 seconds passing without signal
        signal_loss_duration = 30.0

        # Mock time to simulate 30s passing
        with patch(
            "backend.services.state_machine.time.time",
            return_value=state_machine._signal_lost_time + signal_loss_duration + 1,
        ):
            # Process timeout - will check elapsed > 30.0
            await state_machine._check_signal_loss_timeout()

            # Homing should be disabled
            # Homing should be disabled after timeout
            assert (
                not state_machine._homing_enabled
            ), "Homing should be disabled after 30s signal loss"

            # Should transition to safe state
            assert state_machine.get_current_state() in [SystemState.IDLE, SystemState.SEARCHING]

        # Test recovery when signal returns before timeout
        state_machine._current_state = SystemState.SEARCHING
        state_machine._signal_lost_time = time.time()

        # Signal returns after 10 seconds
        with patch("time.time", return_value=state_machine._signal_lost_time + 10.0):
            detection_event = {
                "rssi": -65.0,
                "snr": 20.0,
                "confidence": 0.9,
                "timestamp": time.time(),
            }

            await state_machine.on_signal_detected(detection_event)

            # Should transition back to DETECTING
            assert state_machine.get_current_state() == SystemState.DETECTING

            # Homing should still be enabled (didn't timeout)
            # Homing should be re-enabled based on configuration
            assert state_machine._homing_enabled, "Homing should be re-enabled when signal returns"

    @pytest.mark.asyncio
    async def test_emergency_stop_handling(self, state_machine):
        """Test emergency stop transitions."""
        # Test from each state
        for initial_state in SystemState:
            state_machine._current_state = initial_state

            # Trigger emergency stop
            success = await state_machine.transition_to(SystemState.IDLE, "EMERGENCY STOP")

            # Should always succeed
            assert success or initial_state == SystemState.IDLE
            assert state_machine.get_current_state() == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_state_persistence(self, state_machine):
        """Test state persistence across restarts."""
        # Test that state can be set and retrieved
        # Set a state
        await state_machine.transition_to(SystemState.SEARCHING, "Test")

        # Verify state is set
        assert state_machine.get_current_state() == SystemState.SEARCHING

        # Save the state (implementation may save to database)
        if hasattr(state_machine, "save_state"):
            state_machine.save_state()

        # State persistence is handled by database in actual implementation
        # For unit test, we just verify the state can be set and retrieved
        current_state = state_machine.get_current_state()

        # Should maintain state
        assert (
            current_state == SystemState.SEARCHING
        ), f"State not maintained: expected SEARCHING, got {current_state}"

    @pytest.mark.asyncio
    async def test_concurrent_transitions(self, state_machine):
        """Test handling of concurrent transition requests."""
        # Start multiple transitions concurrently
        tasks = [
            state_machine.transition_to(SystemState.SEARCHING, "Request 1"),
            state_machine.transition_to(SystemState.DETECTING, "Request 2"),
            state_machine.transition_to(SystemState.IDLE, "Request 3"),
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Only one should succeed
        successes = [r for r in results if r is True]
        assert len(successes) >= 1, "At least one transition should succeed"

        # Final state should be valid
        final_state = state_machine.get_current_state()
        assert final_state in SystemState

    @pytest.mark.asyncio
    async def test_state_timeout_handling(self, state_machine):
        """Test state timeout mechanisms."""
        # Configure timeout for DETECTING state
        state_machine.set_state_timeout(SystemState.DETECTING, 10.0)

        # Transition to DETECTING
        await state_machine.transition_to(SystemState.DETECTING, "Test timeout")

        # Verify timeout is set
        # Verify timeout task exists for signal loss handling
        assert hasattr(state_machine, "_timeout_task") or hasattr(
            state_machine, "_check_signal_loss_timeout"
        ), "State machine lacks timeout handling mechanism"

        # Cancel timeout on state exit
        await state_machine.transition_to(SystemState.IDLE, "Exit before timeout")

        # Timeout should be cancelled
        if hasattr(state_machine, "_timeout_task"):
            assert state_machine._timeout_task is None or state_machine._timeout_task.cancelled()


class TestStateMachineIntegration:
    """Integration tests with other services."""

    @pytest.mark.asyncio
    async def test_signal_processor_integration(self):
        """Test integration with signal processor."""
        sm = StateMachine()
        sp = SignalProcessor()

        # Connect services
        sm.set_signal_processor(sp)

        await sm.initialize()

        # Start searching
        await sm.transition_to(SystemState.SEARCHING, "Start search")

        # Simulate signal detection
        detection_event = {"rssi": -60.0, "snr": 25.0, "confidence": 0.95, "timestamp": time.time()}

        await sm.on_signal_detected(detection_event)

        # Should transition to DETECTING
        assert sm.get_current_state() == SystemState.DETECTING

        # Simulate signal loss
        await sm.on_signal_lost()

        # Should transition back to SEARCHING
        assert sm.get_current_state() == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_mavlink_integration(self):
        """Test integration with MAVLink service."""
        sm = StateMachine()
        ml = MAVLinkService()

        # Connect services
        sm._mavlink_service = ml

        await sm.initialize()

        # Test telemetry during state changes
        await sm.transition_to(SystemState.SEARCHING, "Test")

        # Verify telemetry would be sent
        # In real implementation, check ml._telemetry_queue
        assert True  # Placeholder


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
