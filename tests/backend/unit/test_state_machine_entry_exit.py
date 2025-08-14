"""Unit tests for state machine entry/exit actions."""

import asyncio

import pytest

from src.backend.services.state_machine import StateMachine, SystemState


@pytest.fixture
def state_machine():
    """Create a state machine instance for testing."""
    return StateMachine()


@pytest.mark.asyncio
async def test_entry_exit_actions_called_on_transition(state_machine):
    """Test that entry and exit actions are called during state transitions."""
    # Track action calls
    exit_idle_called = False
    entry_searching_called = False

    async def mock_exit_idle():
        nonlocal exit_idle_called
        exit_idle_called = True

    async def mock_entry_searching():
        nonlocal entry_searching_called
        entry_searching_called = True

    # Register custom actions
    state_machine.register_exit_action(SystemState.IDLE, mock_exit_idle)
    state_machine.register_entry_action(SystemState.SEARCHING, mock_entry_searching)

    # Perform transition
    result = await state_machine.transition_to(SystemState.SEARCHING)

    assert result is True
    assert exit_idle_called
    assert entry_searching_called
    assert state_machine.get_current_state() == SystemState.SEARCHING


@pytest.mark.asyncio
async def test_default_actions_registered(state_machine):
    """Test that default entry/exit actions are registered for all states."""
    for state in SystemState:
        assert len(state_machine._entry_actions[state]) > 0
        assert len(state_machine._exit_actions[state]) > 0


@pytest.mark.asyncio
async def test_action_execution_order(state_machine):
    """Test that exit actions run before entry actions."""
    execution_order = []

    async def exit_action():
        execution_order.append("exit")

    async def entry_action():
        execution_order.append("entry")

    # Clear default actions first
    state_machine._exit_actions[SystemState.IDLE] = []
    state_machine._entry_actions[SystemState.SEARCHING] = []

    state_machine.register_exit_action(SystemState.IDLE, exit_action)
    state_machine.register_entry_action(SystemState.SEARCHING, entry_action)

    await state_machine.transition_to(SystemState.SEARCHING)

    assert execution_order == ["exit", "entry"]


@pytest.mark.asyncio
async def test_action_exception_handling(state_machine):
    """Test that exceptions in actions don't prevent state transition."""

    async def failing_exit_action():
        raise Exception("Exit action failed")

    async def failing_entry_action():
        raise Exception("Entry action failed")

    state_machine.register_exit_action(SystemState.IDLE, failing_exit_action)
    state_machine.register_entry_action(SystemState.SEARCHING, failing_entry_action)

    # Should still transition despite exceptions
    result = await state_machine.transition_to(SystemState.SEARCHING)

    assert result is True
    assert state_machine.get_current_state() == SystemState.SEARCHING


@pytest.mark.asyncio
async def test_multiple_actions_per_state(state_machine):
    """Test that multiple actions can be registered and executed for a state."""
    action_calls = []

    async def action1():
        action_calls.append("action1")

    async def action2():
        action_calls.append("action2")

    async def action3():
        action_calls.append("action3")

    # Clear default actions first
    state_machine._entry_actions[SystemState.SEARCHING] = []

    state_machine.register_entry_action(SystemState.SEARCHING, action1)
    state_machine.register_entry_action(SystemState.SEARCHING, action2)
    state_machine.register_entry_action(SystemState.SEARCHING, action3)

    await state_machine.transition_to(SystemState.SEARCHING)

    assert len(action_calls) == 3
    assert "action1" in action_calls
    assert "action2" in action_calls
    assert "action3" in action_calls


@pytest.mark.asyncio
async def test_no_duplicate_action_registration(state_machine):
    """Test that the same action cannot be registered multiple times."""

    async def my_action():
        pass

    state_machine.register_entry_action(SystemState.SEARCHING, my_action)
    state_machine.register_entry_action(SystemState.SEARCHING, my_action)

    # Should only be registered once
    assert state_machine._entry_actions[SystemState.SEARCHING].count(my_action) == 1


@pytest.mark.asyncio
async def test_action_timing_measurement(state_machine):
    """Test that action execution time is measured correctly."""

    async def slow_action():
        await asyncio.sleep(0.05)  # 50ms delay

    state_machine.register_exit_action(SystemState.IDLE, slow_action)

    duration = await state_machine._execute_exit_actions(SystemState.IDLE)

    # Should be at least 50ms
    assert duration >= 50.0


@pytest.mark.asyncio
async def test_searching_exit_pauses_pattern(state_machine):
    """Test that exiting SEARCHING state pauses active search pattern."""
    from src.backend.services.search_pattern_generator import PatternType, SearchPattern

    # Create a mock search pattern
    pattern = SearchPattern(
        id="test-pattern",
        pattern_type=PatternType.SPIRAL,
        center_lat=0.0,
        center_lon=0.0,
        size_meters=100,
        spacing_meters=10,
        waypoints=[],
        total_waypoints=10,
        state="IDLE",
    )

    state_machine.set_search_pattern(pattern)

    # Start searching and pattern execution
    await state_machine.transition_to(SystemState.SEARCHING)
    await state_machine.start_search_pattern()

    # Transition to DETECTING should pause the pattern
    await state_machine.transition_to(SystemState.DETECTING)

    from src.backend.services.state_machine import SearchSubstate

    assert state_machine._search_substate == SearchSubstate.PAUSED


@pytest.mark.asyncio
async def test_idle_entry_releases_resources(state_machine):
    """Test that entering IDLE state releases resources."""
    from src.backend.services.state_machine import SearchSubstate

    # Set up some state
    state_machine._search_substate = SearchSubstate.EXECUTING
    state_machine._current_waypoint_index = 5

    # Transition to IDLE
    await state_machine.transition_to(SystemState.IDLE)

    # Resources should be released
    assert state_machine._search_substate == SearchSubstate.IDLE
    assert state_machine._current_waypoint_index == 0


@pytest.mark.asyncio
async def test_no_actions_on_same_state_transition(state_machine):
    """Test that transitioning to the same state doesn't execute actions."""
    action_called = False

    async def track_action():
        nonlocal action_called
        action_called = True

    # Already in IDLE state
    state_machine.register_exit_action(SystemState.IDLE, track_action)
    state_machine.register_entry_action(SystemState.IDLE, track_action)

    result = await state_machine.transition_to(SystemState.IDLE)

    assert result is True
    assert not action_called  # Actions should not be called
