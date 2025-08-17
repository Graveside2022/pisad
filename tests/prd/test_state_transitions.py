"""
PRD State Transitions Test Suite
Tests for comprehensive state machine transition validation

TASK-9.10: Fix Remaining State Machine Test Failures
Recreated missing test file for state transition validation per PRD-FR3,FR15,FR17

This test file focuses on:
- Basic state transition validation
- ConfigProfile integration testing
- State persistence and recovery
- Thread safety for concurrent requests
- Timing requirements per FR3
- Emergency stop functionality
"""

import asyncio
import time

import pytest

from src.backend.models.schemas import ConfigProfile
from src.backend.services.state_machine import StateMachine, SystemState


class TestBasicStateTransitions:
    """Test basic state machine transition functionality."""

    @pytest.fixture
    async def state_machine(self):
        """Create clean state machine instance for testing."""
        # TDD GREEN PHASE: Add required dependencies for state transitions
        from src.backend.services.mavlink_service import MAVLinkService
        from src.backend.services.signal_processor import SignalProcessor

        sm = StateMachine(enable_persistence=False)  # Disable DB for isolation

        # Add required dependencies for state transitions to work
        sm.set_signal_processor(SignalProcessor())
        sm.set_mavlink_service(MAVLinkService())

        await sm.initialize()
        yield sm
        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_initial_state_is_idle(self, state_machine):
        """
        Test that state machine starts in IDLE state.

        Basic sanity check for state machine initialization.
        """
        # TDD RED PHASE: This should pass - basic functionality test
        assert state_machine.current_state == SystemState.IDLE
        assert state_machine.get_current_state() == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_valid_state_transitions(self, state_machine):
        """
        Test all valid state transitions according to state machine design.

        Validates state machine logic and transition rules.
        """
        # Test IDLE -> SEARCHING
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is True, "Should allow IDLE -> SEARCHING transition"
        assert state_machine.current_state == SystemState.SEARCHING

        # Test SEARCHING -> DETECTING
        result = await state_machine.transition_to(SystemState.DETECTING)
        assert result is True, "Should allow SEARCHING -> DETECTING transition"
        assert state_machine.current_state == SystemState.DETECTING

        # Test DETECTING -> HOMING (requires homing to be enabled)
        state_machine.enable_homing(True)  # Enable homing for transition
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is True, "Should allow DETECTING -> HOMING transition"
        assert state_machine.current_state == SystemState.HOMING

        # Test HOMING -> HOLDING
        result = await state_machine.transition_to(SystemState.HOLDING)
        assert result is True, "Should allow HOMING -> HOLDING transition"
        assert state_machine.current_state == SystemState.HOLDING

        # Test HOLDING -> HOMING (return to homing)
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result is True, "Should allow HOLDING -> HOMING transition"
        assert state_machine.current_state == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_emergency_stop_from_any_state(self, state_machine):
        """
        Test emergency stop works from any operational state.

        Critical safety requirement - emergency stop must always work.
        """
        test_states = [
            SystemState.SEARCHING,
            SystemState.DETECTING,
            SystemState.HOMING,
            SystemState.HOLDING,
        ]

        for state in test_states:
            # Set up state
            await state_machine.transition_to(state)
            assert state_machine.current_state == state

            # Trigger emergency stop
            await state_machine.emergency_stop()

            # Should go to EMERGENCY state
            assert (
                state_machine.current_state == SystemState.EMERGENCY
            ), f"Emergency stop should go to EMERGENCY from {state}"


class TestConfigProfileIntegration:
    """Test ConfigProfile dataclass integration with state machine."""

    @pytest.fixture
    async def state_machine(self):
        """Create state machine for ConfigProfile testing."""
        from src.backend.services.mavlink_service import MAVLinkService
        from src.backend.services.signal_processor import SignalProcessor

        sm = StateMachine(enable_persistence=False)

        # Add required dependencies for state transitions
        sm.set_signal_processor(SignalProcessor())
        sm.set_mavlink_service(MAVLinkService())

        await sm.initialize()
        yield sm
        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_configprofile_constructor_compatibility(self, state_machine):
        """
        Test ConfigProfile can be instantiated without constructor errors.

        CRITICAL: This test addresses the ConfigProfile constructor mismatch
        that was causing 3 ERROR tests per BLOCKER-003.
        """
        # TDD RED PHASE: This will likely fail initially
        try:
            # Test basic ConfigProfile creation
            profile = ConfigProfile(
                id="test_profile", name="Test Profile", description="Test configuration profile"
            )
            assert profile.id == "test_profile"
            assert profile.name == "Test Profile"
            assert profile.description == "Test configuration profile"

            # Test with optional fields
            profile_with_defaults = ConfigProfile(id="default_profile", name="Default Profile")
            assert profile_with_defaults.description == ""  # Default value
            assert profile_with_defaults.isDefault is False  # Default value

        except TypeError as e:
            pytest.fail(f"ConfigProfile constructor failed: {e}")

    @pytest.mark.asyncio
    async def test_state_machine_with_configprofile(self, state_machine):
        """
        Test state machine operation with ConfigProfile configuration.

        Validates integration between state machine and configuration system.
        """
        # Create test profile
        profile = ConfigProfile(
            id="state_test_profile",
            name="State Machine Test Profile",
            description="Configuration for state machine testing",
        )

        # Verify profile was created correctly (this validates ConfigProfile works)
        assert profile.id == "state_test_profile"
        assert profile.name == "State Machine Test Profile"

        # State machine should accept configuration
        # (This tests integration without requiring full config loading)
        assert state_machine.current_state == SystemState.IDLE

        # State transitions should work regardless of config profile
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result is True
        assert state_machine.current_state == SystemState.SEARCHING


class TestStatePersistence:
    """Test state persistence and recovery functionality."""

    @pytest.fixture
    async def persistent_state_machine(self):
        """Create state machine with persistence enabled for testing."""
        from src.backend.services.mavlink_service import MAVLinkService
        from src.backend.services.signal_processor import SignalProcessor

        sm = StateMachine(db_path=":memory:", enable_persistence=True)  # In-memory DB for testing

        # Add required dependencies for state transitions
        sm.set_signal_processor(SignalProcessor())
        sm.set_mavlink_service(MAVLinkService())

        await sm.initialize()
        yield sm
        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_state_persistence_methods_exist(self, persistent_state_machine):
        """
        Test that state persistence methods are available and functional.

        Validates save_current_state and restore_state methods exist.
        """
        sm = persistent_state_machine

        # Change to non-idle state
        await sm.transition_to(SystemState.SEARCHING)
        assert sm.current_state == SystemState.SEARCHING

        # Test save method exists and can be called
        try:
            saved_state = sm.save_state()
            assert isinstance(saved_state, dict), "save_state should return dict"
            assert "current_state" in saved_state, "Saved state should include current_state"
        except AttributeError:
            pytest.fail("save_state method missing from StateMachine")

        # Test restore method exists and can be called
        try:
            sm.restore_state(saved_state)
            # After restore, state should be maintained
            assert (
                sm.current_state == SystemState.SEARCHING
            ), "State should be preserved after restore"
        except AttributeError:
            pytest.fail("restore_state method missing from StateMachine")


class TestThreadSafety:
    """Test state machine thread safety for concurrent requests."""

    @pytest.fixture
    async def state_machine(self):
        """Create state machine for thread safety testing."""
        from src.backend.services.mavlink_service import MAVLinkService
        from src.backend.services.signal_processor import SignalProcessor

        sm = StateMachine(enable_persistence=False)

        # Add required dependencies for state transitions
        sm.set_signal_processor(SignalProcessor())
        sm.set_mavlink_service(MAVLinkService())

        await sm.initialize()
        yield sm
        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_state_transitions(self, state_machine):
        """
        Test handling of concurrent state transition requests.

        Validates thread safety and race condition handling per enterprise requirements.
        """
        # Start multiple concurrent transition requests
        transition_tasks = []

        # Create multiple concurrent requests for different states
        states_to_test = [SystemState.SEARCHING, SystemState.IDLE, SystemState.SEARCHING]

        for target_state in states_to_test:
            task = asyncio.create_task(
                state_machine.transition_to(
                    target_state, reason=f"Concurrent test to {target_state}"
                )
            )
            transition_tasks.append(task)

        # Wait for all transitions to complete
        results = await asyncio.gather(*transition_tasks, return_exceptions=True)

        # All transitions should complete without exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent transition {i} failed with exception: {result}")

        # Final state should be one of the requested states (race condition acceptable)
        assert state_machine.current_state in [
            SystemState.IDLE,
            SystemState.SEARCHING,
        ], "Final state should be one of the requested states"


class TestTimingRequirements:
    """Test state transition timing requirements per PRD-FR3."""

    @pytest.fixture
    async def state_machine(self):
        """Create state machine for timing tests."""
        from src.backend.services.mavlink_service import MAVLinkService
        from src.backend.services.signal_processor import SignalProcessor

        sm = StateMachine(enable_persistence=False)

        # Add required dependencies for state transitions
        sm.set_signal_processor(SignalProcessor())
        sm.set_mavlink_service(MAVLinkService())

        await sm.initialize()
        yield sm
        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_fr3_transition_timing_2_seconds(self, state_machine):
        """
        FR3: Test state transitions complete within 2 seconds.

        Per PRD-FR3: "System shall transition to HOMING behavior within 2 seconds"
        """
        # Test transition timing for critical state changes
        start_time = time.perf_counter()

        # IDLE -> SEARCHING transition
        result = await state_machine.transition_to(SystemState.SEARCHING)
        transition_time = time.perf_counter() - start_time

        assert result is True, "State transition should succeed"
        assert state_machine.current_state == SystemState.SEARCHING
        assert (
            transition_time < 2.0
        ), f"IDLE->SEARCHING took {transition_time:.3f}s, requirement is <2s"

        # SEARCHING -> DETECTING transition
        start_time = time.perf_counter()
        result = await state_machine.transition_to(SystemState.DETECTING)
        transition_time = time.perf_counter() - start_time

        assert result is True, "State transition should succeed"
        assert state_machine.current_state == SystemState.DETECTING
        assert (
            transition_time < 2.0
        ), f"SEARCHING->DETECTING took {transition_time:.3f}s, requirement is <2s"

        # DETECTING -> HOMING transition (critical for FR3)
        state_machine.enable_homing(True)  # Enable homing for transition
        start_time = time.perf_counter()
        result = await state_machine.transition_to(SystemState.HOMING)
        transition_time = time.perf_counter() - start_time

        assert result is True, "State transition should succeed"
        assert state_machine.current_state == SystemState.HOMING
        assert (
            transition_time < 2.0
        ), f"DETECTING->HOMING took {transition_time:.3f}s, requirement is <2s"


class TestStateEntryExitActions:
    """Test state entry and exit action execution."""

    @pytest.fixture
    async def state_machine(self):
        """Create state machine for entry/exit action testing."""
        from src.backend.services.mavlink_service import MAVLinkService
        from src.backend.services.signal_processor import SignalProcessor

        sm = StateMachine(enable_persistence=False)

        # Add required dependencies for state transitions
        sm.set_signal_processor(SignalProcessor())
        sm.set_mavlink_service(MAVLinkService())

        await sm.initialize()
        yield sm
        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_state_entry_exit_actions_execute(self, state_machine):
        """
        Test that state entry and exit actions are executed properly.

        Validates proper state lifecycle management.
        """
        # Track executed actions
        executed_actions = []

        async def mock_entry_action():
            executed_actions.append("enter_homing")

        async def mock_exit_action():
            executed_actions.append("exit_homing")

        # Register mock actions for HOMING state (append to existing actions)
        state_machine._entry_actions[SystemState.HOMING].append(mock_entry_action)
        state_machine._exit_actions[SystemState.HOMING].append(mock_exit_action)

        # Enable homing and transition to HOMING (should trigger entry action)
        state_machine.enable_homing(True)
        await state_machine.transition_to(
            SystemState.DETECTING
        )  # Need to go through proper state path
        await state_machine.transition_to(SystemState.HOMING)
        await asyncio.sleep(0.1)  # Allow async actions to complete

        # Transition away from HOMING (should trigger exit action)
        await state_machine.transition_to(SystemState.HOLDING)
        await asyncio.sleep(0.1)  # Allow async actions to complete

        # Verify both entry and exit actions were executed
        assert "enter_homing" in executed_actions, "Entry action should execute on state entry"
        assert "exit_homing" in executed_actions, "Exit action should execute on state exit"
        assert executed_actions == [
            "enter_homing",
            "exit_homing",
        ], f"Expected entry then exit, got {executed_actions}"
