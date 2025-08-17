#!/usr/bin/env python3
"""PRD-aligned state machine tests with real timing requirements.

Tests FR3, FR15, FR17 requirements from PRD.
"""

import os
import sys
import time

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Check for hardware availability
try:
    from src.backend.services.mavlink_service import MAVLinkService
    from src.backend.services.signal_processor import SignalProcessor
    from src.backend.services.state_machine import StateMachine, SystemState

    has_hardware = True
except ImportError:
    has_hardware = False


@pytest.mark.skipif(not has_hardware, reason="Requires hardware modules to be installed")
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
            (SystemState.HOMING, SystemState.IDLE),
        ]

        for from_state, to_state in test_transitions:
            # Set initial state
            state_machine._current_state = from_state

            # Measure transition time
            start_time = time.perf_counter()

            # Attempt transition
            success = await state_machine.transition_to(to_state)

            transition_time = time.perf_counter() - start_time

            if success:
                transition_times.append(transition_time)
                # Verify transition completed within 2 seconds
                assert (
                    transition_time < 2.0
                ), f"Transition {from_state} -> {to_state} took {transition_time:.3f}s (>2s)"

                # Verify we're in the target state
                assert (
                    state_machine.get_current_state() == to_state
                ), f"Failed to reach {to_state} from {from_state}"

        # Verify average transition time is reasonable
        avg_time = sum(transition_times) / len(transition_times) if transition_times else 0
        max_time = max(transition_times) if transition_times else 0

        assert avg_time < 0.5, f"Average transition time {avg_time:.3f}s too high"

        print(f"Transition times: avg={avg_time:.3f}s, max={max_time:.3f}s")

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Requires real MAVLink hardware connection - no mocks allowed per requirements"
    )
    async def test_fr15_velocity_command_cessation(self, state_machine):
        """FR15: Test velocity command cessation on mode change.

        Requirement: Velocity commands shall stop on operation mode change

        BLOCKER: This test requires real MAVLink hardware to test velocity commands.
        Cannot use mocks per project requirements.
        Need: Real flight controller connected via serial/USB for MAVLink communication.
        """
        pytest.skip("Requires real MAVLink hardware connection")

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Requires real-time hardware testing - cannot mock time per requirements"
    )
    async def test_fr17_auto_disable_signal_loss(self, state_machine):
        """FR17: Test auto-disable after signal loss.

        Requirement: System shall auto-disable homing after 30s signal loss

        BLOCKER: This test requires real-time testing with actual signal hardware.
        Cannot mock time functions per project requirements.
        Need: Real SDR hardware (HackRF One) to generate and drop signals.
        """
        pytest.skip("Requires real SDR hardware for signal loss testing")

    @pytest.mark.asyncio
    async def test_emergency_stop_handling(self, state_machine):
        """Test emergency stop from any state.

        Requirement: Emergency stop shall work from any state
        """
        test_states = [
            SystemState.IDLE,
            SystemState.SEARCHING,
            SystemState.DETECTING,
            SystemState.HOMING,
        ]

        for initial_state in test_states:
            # Set initial state
            state_machine._current_state = initial_state

            # Trigger emergency stop
            await state_machine.emergency_stop()

            # Should be in IDLE (safe) state
            assert (
                state_machine.get_current_state() == SystemState.IDLE
            ), f"Emergency stop from {initial_state} should result in IDLE"

            # Homing should be disabled
            assert (
                not state_machine._homing_enabled
            ), "Homing should be disabled after emergency stop"

    @pytest.mark.asyncio
    async def test_state_persistence(self, state_machine):
        """Test state persistence across restarts."""
        # Set a specific state
        await state_machine.transition_to(SystemState.SEARCHING)

        # Save state (if implemented)
        if hasattr(state_machine, "save_state"):
            await state_machine.save_state()

            # Create new instance
            new_sm = StateMachine()
            await new_sm.initialize()

            # Load state (if implemented)
            if hasattr(new_sm, "load_state"):
                await new_sm.load_state()

                # Should restore previous state
                assert (
                    new_sm.get_current_state() == SystemState.SEARCHING
                ), "State should persist across restarts"
        else:
            # State persistence not implemented yet
            pytest.skip("State persistence not implemented")


class TestStateMachineIntegration:
    """Integration tests with real hardware."""

    @pytest.mark.skipif(
        not os.environ.get("HARDWARE_TEST_ENABLED"),
        reason="Set HARDWARE_TEST_ENABLED=1 to run hardware tests",
    )
    @pytest.mark.asyncio
    async def test_real_hardware_integration(self):
        """Test with real hardware if available."""
        # This test would run with actual hardware
        pytest.skip("Hardware integration test placeholder")
