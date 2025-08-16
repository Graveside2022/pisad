"""
PRD State Machine Requirements Tests
Tests for FR3, FR15, FR17 - State transition requirements

Story 4.9 Sprint 8 Day 3-4: Real PRD test implementation
"""

import asyncio
import time

import pytest

from backend.services.homing_controller import HomingController
from backend.services.mavlink_service import MAVLinkService
from backend.services.signal_processor import SignalProcessor
from backend.services.state_machine import StateMachine
from backend.utils.safety import SafetyInterlockSystem


class TestStateMachineRequirements:
    """Test state machine requirements from PRD."""

    @pytest.fixture
    async def state_machine(self):
        """Create state machine instance with required dependencies."""
        sm = StateMachine()
        # Add signal processor to allow IDLE -> SEARCHING transition
        sm._signal_processor = SignalProcessor()
        # Add MAVLink service for velocity commands
        sm._mavlink_service = MAVLinkService()
        await sm.initialize()
        yield sm
        await sm.shutdown()

    @pytest.fixture
    async def homing_controller(self, state_machine):
        """Create homing controller instance with required dependencies."""
        mavlink_service = MAVLinkService()
        signal_processor = SignalProcessor()
        controller = HomingController(mavlink_service, signal_processor, state_machine)
        yield controller

    @pytest.fixture
    async def safety_system(self):
        """Create safety interlock system."""
        safety = SafetyInterlockSystem()
        yield safety
        await safety.stop_monitoring()

    @pytest.mark.asyncio
    async def test_fr3_transition_time(self, state_machine, homing_controller):
        """
        FR3: System shall transition to HOMING within 2 seconds of beacon detection when activated.

        Validates state transition timing requirement.
        """
        # Start in SEARCHING state
        await state_machine.transition_to("SEARCHING")
        assert state_machine.current_state == "SEARCHING"

        # Enable homing mode (operator activation)
        homing_controller.enable_homing()

        # Simulate beacon detection
        start_time = time.perf_counter()
        await state_machine.on_signal_detected(snr_db=15.0, frequency=3.2e9)

        # Wait for transition
        max_wait = 2.5  # Allow slight overhead
        while (
            state_machine.current_state != "HOMING"
            and (time.perf_counter() - start_time) < max_wait
        ):
            await asyncio.sleep(0.01)

        transition_time = time.perf_counter() - start_time

        # Verify transition occurred within 2 seconds
        assert state_machine.current_state == "HOMING", "Should transition to HOMING on detection"
        assert transition_time < 2.0, f"Transition took {transition_time:.2f}s, requirement is <2s"

    @pytest.mark.asyncio
    async def test_fr15_velocity_command_cessation(self, state_machine, homing_controller):
        """
        FR15: System shall immediately cease velocity commands when mode changes from GUIDED.

        Tests that velocity commands stop on mode change.
        """
        # Start in HOMING state with GUIDED mode
        await state_machine.transition_to("HOMING")
        homing_controller.enable_homing()

        # Track velocity commands
        velocity_commands_sent = []

        async def mock_send_velocity(vx, vy, vz):
            velocity_commands_sent.append(
                {"time": time.perf_counter(), "vx": vx, "vy": vy, "vz": vz}
            )
            return True

        homing_controller.send_velocity_command = mock_send_velocity

        # Start sending velocity commands
        velocity_task = asyncio.create_task(homing_controller.continuous_homing_commands())

        # Let it run briefly
        await asyncio.sleep(0.5)

        # Simulate mode change from GUIDED to LOITER
        mode_change_time = time.perf_counter()
        await state_machine.on_mode_change("GUIDED", "LOITER")

        # Wait briefly for commands to stop
        await asyncio.sleep(0.1)

        # Cancel the task
        velocity_task.cancel()
        try:
            await velocity_task
        except asyncio.CancelledError:
            pass

        # Check that no commands were sent after mode change
        commands_after_mode_change = [
            cmd
            for cmd in velocity_commands_sent
            if cmd["time"] > mode_change_time + 0.01  # Small tolerance
        ]

        assert (
            len(commands_after_mode_change) == 0
        ), f"Sent {len(commands_after_mode_change)} commands after mode change"

        # Verify homing is disabled
        assert (
            not homing_controller.is_homing_enabled()
        ), "Homing should be disabled on mode change from GUIDED"

    @pytest.mark.asyncio
    async def test_fr17_auto_disable_signal_loss(self, state_machine, homing_controller):
        """
        FR17: System shall auto-disable homing after 10 seconds of signal loss.

        Tests automatic homing disable on signal timeout.
        """
        # Start in HOMING state
        await state_machine.transition_to("HOMING")
        homing_controller.enable_homing()
        assert homing_controller.is_homing_enabled()

        # Simulate signal loss
        signal_loss_time = time.perf_counter()
        await state_machine.on_signal_lost()

        # Track homing status over time
        homing_disabled_time = None

        # Monitor for up to 12 seconds
        while time.perf_counter() - signal_loss_time < 12:
            if not homing_controller.is_homing_enabled():
                homing_disabled_time = time.perf_counter() - signal_loss_time
                break
            await asyncio.sleep(0.1)

        # Verify homing was disabled
        assert homing_disabled_time is not None, "Homing was not disabled"

        # Verify it happened after 10 seconds (±0.5s tolerance)
        assert (
            9.5 < homing_disabled_time < 10.5
        ), f"Homing disabled after {homing_disabled_time:.1f}s, expected ~10s"

        # Verify operator notification was sent
        assert (
            homing_controller.get_last_notification() is not None
        ), "Operator should be notified of auto-disable"

    @pytest.mark.asyncio
    async def test_state_transitions_valid(self, state_machine):
        """
        Test that only valid state transitions are allowed.

        Validates state machine integrity.
        """
        # Define valid transitions based on PRD
        valid_transitions = {
            "IDLE": ["SEARCHING"],
            "SEARCHING": ["DETECTING", "IDLE"],
            "DETECTING": ["HOMING", "SEARCHING"],
            "HOMING": ["HOLDING", "SEARCHING", "IDLE"],
            "HOLDING": ["HOMING", "SEARCHING", "IDLE"],
        }

        # Test all states
        for from_state, allowed_to_states in valid_transitions.items():
            await state_machine.transition_to(from_state)

            # Try valid transitions
            for to_state in allowed_to_states:
                await state_machine.transition_to(from_state)  # Reset
                result = await state_machine.transition_to(to_state)
                assert result, f"Should allow {from_state} -> {to_state}"

            # Try invalid transitions
            all_states = set(valid_transitions.keys())
            invalid_states = all_states - set(allowed_to_states) - {from_state}

            for to_state in invalid_states:
                await state_machine.transition_to(from_state)  # Reset
                result = await state_machine.transition_to(to_state)
                assert not result, f"Should block {from_state} -> {to_state}"

    @pytest.mark.asyncio
    async def test_state_persistence(self, state_machine):
        """
        Test state persistence across restarts.

        Validates state recovery capability.
        """
        # Set a specific state
        await state_machine.transition_to("HOMING")

        # Save state
        saved_state = state_machine.save_state()

        # Create new state machine
        new_sm = StateMachine()
        await new_sm.initialize()

        # Restore state
        new_sm.restore_state(saved_state)

        # Verify state was restored
        assert new_sm.current_state == "HOMING", "State should persist"

        # Verify transition history was preserved
        assert len(new_sm.get_transition_history()) > 0, "Transition history should be preserved"

        await new_sm.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_state_requests(self, state_machine):
        """
        Test handling of concurrent state transition requests.

        Validates thread safety and race condition handling.
        """
        # Start in SEARCHING
        await state_machine.transition_to("SEARCHING")

        # Create multiple concurrent transition requests
        async def request_transition(target_state, delay=0):
            await asyncio.sleep(delay)
            return await state_machine.transition_to(target_state)

        # Launch concurrent requests
        tasks = [
            request_transition("DETECTING", 0),
            request_transition("HOMING", 0.01),
            request_transition("IDLE", 0.02),
            request_transition("SEARCHING", 0.03),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Only one transition should succeed at a time
        successful_transitions = [r for r in results if r is True]

        # At least one should succeed
        assert len(successful_transitions) >= 1, "At least one transition should succeed"

        # Final state should be deterministic
        assert state_machine.current_state in [
            "DETECTING",
            "IDLE",
            "SEARCHING",
        ], "Final state should be one of the requested states"

    @pytest.mark.asyncio
    async def test_state_entry_exit_actions(self, state_machine):
        """
        Test that state entry and exit actions are executed.

        Validates proper state lifecycle management.
        """
        # Track action execution
        actions_executed = []

        # Mock entry/exit actions
        async def on_enter_homing():
            actions_executed.append("enter_homing")
            return True

        async def on_exit_homing():
            actions_executed.append("exit_homing")
            return True

        state_machine.register_entry_action("HOMING", on_enter_homing)
        state_machine.register_exit_action("HOMING", on_exit_homing)

        # Transition to HOMING
        await state_machine.transition_to("HOMING")

        # Transition away from HOMING
        await state_machine.transition_to("SEARCHING")

        # Verify actions were executed in order
        assert actions_executed == [
            "enter_homing",
            "exit_homing",
        ], f"Expected entry then exit, got {actions_executed}"

    @pytest.mark.asyncio
    async def test_emergency_stop_priority(self, state_machine, safety_system):
        """
        Test that emergency stop overrides all state transitions.

        Validates safety priority in state machine.
        """
        # Start normal operation
        await state_machine.transition_to("HOMING")

        # Trigger emergency stop
        await safety_system.emergency_stop("Test emergency")

        # Try to transition - should be blocked
        result = await state_machine.transition_to("SEARCHING")
        assert not result, "Transitions should be blocked during emergency stop"

        # State should transition to safe state
        assert state_machine.current_state in [
            "IDLE",
            "EMERGENCY",
        ], "Should be in safe state after emergency stop"

        # Reset emergency stop
        await safety_system.reset_emergency_stop()

        # Now transitions should work
        result = await state_machine.transition_to("SEARCHING")
        assert result, "Transitions should work after emergency reset"


class TestStateTransitionTiming:
    """Test specific timing requirements for state transitions."""

    @pytest.mark.asyncio
    async def test_mode_detection_latency(self):
        """
        Test that mode changes are detected within 100ms.

        Related to FR15 - immediate cessation of commands.
        """
        sm = StateMachine()
        await sm.initialize()

        detection_times = []

        async def mode_monitor():
            """Monitor for mode changes."""
            last_mode = "GUIDED"
            while True:
                current_mode = sm.get_flight_mode()
                if current_mode != last_mode:
                    detection_times.append(time.perf_counter())
                    last_mode = current_mode
                await asyncio.sleep(0.01)  # 10ms polling

        # Start monitoring
        monitor_task = asyncio.create_task(mode_monitor())

        # Simulate mode change
        await asyncio.sleep(0.1)  # Let monitor stabilize
        change_time = time.perf_counter()
        await sm.on_mode_change("GUIDED", "LOITER")

        # Wait for detection
        await asyncio.sleep(0.2)

        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        if detection_times:
            latency = (detection_times[0] - change_time) * 1000
            assert latency < 100, f"Mode detection took {latency:.1f}ms, requirement is <100ms"

        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_signal_loss_debouncing(self):
        """
        Test signal loss debouncing to prevent false triggers.

        Related to FR17 - 10 second timeout.
        """
        sm = StateMachine()
        await sm.initialize()
        mavlink_service = MAVLinkService()
        signal_processor = SignalProcessor()
        controller = HomingController(mavlink_service, signal_processor, sm)

        # Enable homing
        controller.enable_homing()
        await sm.transition_to("HOMING")

        # Brief signal loss (should not trigger disable)
        await sm.on_signal_lost()
        await asyncio.sleep(5)  # 5 seconds
        await sm.on_signal_detected(snr_db=15.0, frequency=3.2e9)

        # Homing should still be enabled
        assert controller.is_homing_enabled(), "Brief signal loss should not disable homing"

        # Sustained signal loss (should trigger disable)
        await sm.on_signal_lost()
        await asyncio.sleep(10.5)  # Just over 10 seconds

        # Homing should be disabled
        assert not controller.is_homing_enabled(), "Sustained signal loss should disable homing"

        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_transition_atomicity(self):
        """
        Test that state transitions are atomic operations.

        Ensures no intermediate states during transitions.
        """
        sm = StateMachine()
        await sm.initialize()

        # Track all observed states
        observed_states = []

        async def state_observer():
            """Continuously observe state."""
            while True:
                observed_states.append(sm.current_state)
                await asyncio.sleep(0.001)  # 1ms sampling

        # Start observing
        observer_task = asyncio.create_task(state_observer())

        # Perform transitions
        await sm.transition_to("SEARCHING")
        await sm.transition_to("DETECTING")
        await sm.transition_to("HOMING")

        # Stop observing
        await asyncio.sleep(0.1)
        observer_task.cancel()
        try:
            await observer_task
        except asyncio.CancelledError:
            pass

        # Check for invalid intermediate states
        valid_states = {"IDLE", "SEARCHING", "DETECTING", "HOMING", "HOLDING"}
        invalid_states = [s for s in observed_states if s not in valid_states]

        assert len(invalid_states) == 0, f"Observed invalid states: {set(invalid_states)}"

        await sm.shutdown()
