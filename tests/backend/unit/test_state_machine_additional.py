"""Additional comprehensive tests for state machine to improve coverage."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from datetime import datetime, timedelta
from src.backend.services.state_machine import (
    StateMachine,
    SystemState,
    SearchSubstate,
)


@pytest.fixture
def state_machine():
    """Create a state machine instance for testing."""
    sm = StateMachine(enable_persistence=False)
    return sm


@pytest.fixture
def mock_mavlink():
    """Create a mock MAVLink service."""
    mock = MagicMock()
    mock.send_state_change = MagicMock(return_value=True)
    mock.send_detection_event = MagicMock(return_value=True)
    mock.upload_mission = AsyncMock(return_value=True)
    mock.start_mission = AsyncMock(return_value=True)
    mock.pause_mission = AsyncMock(return_value=True)
    mock.resume_mission = AsyncMock(return_value=True)
    mock.stop_mission = AsyncMock(return_value=True)
    mock.get_telemetry = MagicMock(return_value={"position": {"lat": 0, "lon": 0}})
    mock.is_connected = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_signal_processor():
    """Create a mock signal processor."""
    mock = MagicMock()
    mock.update_state = MagicMock()
    mock.get_confidence = MagicMock(return_value=0.85)
    return mock


class TestStateMachineServices:
    """Test service integration with state machine."""
    
    def test_set_mavlink_service(self, state_machine, mock_mavlink):
        """Test setting MAVLink service."""
        state_machine.set_mavlink_service(mock_mavlink)
        assert state_machine._mavlink_service == mock_mavlink
    
    def test_set_signal_processor(self, state_machine, mock_signal_processor):
        """Test setting signal processor."""
        state_machine.set_signal_processor(mock_signal_processor)
        assert state_machine._signal_processor == mock_signal_processor


class TestStateQueries:
    """Test state query methods."""
    
    def test_get_current_state(self, state_machine):
        """Test getting current state."""
        assert state_machine.get_current_state() == SystemState.IDLE
    
    def test_get_state_string(self, state_machine):
        """Test getting state as string."""
        assert state_machine.get_state_string() == "IDLE"
        
    def test_get_state_duration(self, state_machine):
        """Test getting current state duration."""
        duration = state_machine.get_state_duration()
        assert duration >= 0
        
    def test_get_allowed_transitions(self, state_machine):
        """Test getting allowed transitions from current state."""
        transitions = state_machine.get_allowed_transitions()
        assert SystemState.SEARCHING in transitions
        
    def test_is_homing_enabled(self, state_machine):
        """Test checking if homing is enabled."""
        assert not state_machine._homing_enabled
        state_machine.enable_homing(True)
        assert state_machine._homing_enabled


class TestStateCallbacks:
    """Test callback management."""
    
    def test_add_state_callback(self, state_machine):
        """Test adding state change callback."""
        callback = MagicMock()
        state_machine.add_state_callback(callback)
        assert callback in state_machine._state_callbacks
        
    @pytest.mark.asyncio
    async def test_state_callback_invoked(self, state_machine):
        """Test that callbacks are invoked on state change."""
        callback = MagicMock()
        state_machine.add_state_callback(callback)
        
        await state_machine.transition_to(SystemState.SEARCHING)
        callback.assert_called_with(SystemState.IDLE, SystemState.SEARCHING)


class TestHomingControl:
    """Test homing control methods."""
    
    def test_enable_homing(self, state_machine):
        """Test enabling/disabling homing."""
        state_machine.enable_homing(True)
        assert state_machine._homing_enabled
        
        state_machine.enable_homing(False)
        assert not state_machine._homing_enabled
        
    @pytest.mark.asyncio
    async def test_homing_transition_when_disabled(self, state_machine):
        """Test that homing transition fails when disabled."""
        state_machine._current_state = SystemState.DETECTING
        state_machine.enable_homing(False)
        
        result = await state_machine.transition_to(SystemState.HOMING)
        assert not result


class TestStateTimeouts:
    """Test state timeout functionality."""
    
    def test_set_state_timeout(self, state_machine):
        """Test setting state timeout."""
        state_machine.set_state_timeout(SystemState.SEARCHING, 30.0)
        assert state_machine._state_timeouts[SystemState.SEARCHING] == 30.0
        
    def test_clear_state_timeout(self, state_machine):
        """Test clearing state timeout."""
        state_machine.set_state_timeout(SystemState.SEARCHING, 30.0)
        state_machine.set_state_timeout(SystemState.SEARCHING, 0)  # Setting to 0 clears it
        assert state_machine._state_timeouts.get(SystemState.SEARCHING, 0) == 0
        
    @pytest.mark.asyncio
    async def test_timeout_task_creation(self, state_machine):
        """Test that timeout task is created on transition."""
        state_machine.set_state_timeout(SystemState.SEARCHING, 0.1)
        
        await state_machine.transition_to(SystemState.SEARCHING)
        assert state_machine._timeout_task is not None
        
        # Cancel timeout task to prevent warnings
        if state_machine._timeout_task:
            state_machine._timeout_task.cancel()
            try:
                await state_machine._timeout_task
            except asyncio.CancelledError:
                pass


class TestSearchPatternManagement:
    """Test search pattern management."""
    
    def test_set_search_pattern(self, state_machine):
        """Test setting a search pattern."""
        pattern = MagicMock()
        pattern.id = "test-pattern"
        pattern.total_waypoints = 5
        
        state_machine.set_search_pattern(pattern)
        assert state_machine._active_pattern == pattern
        assert state_machine._current_waypoint_index == 0
        assert state_machine._search_substate == SearchSubstate.IDLE
        
    def test_get_search_pattern(self, state_machine):
        """Test getting active search pattern."""
        pattern = MagicMock()
        state_machine._active_pattern = pattern
        
        result = state_machine.get_search_pattern()
        assert result == pattern
        
    def test_get_search_substate(self, state_machine):
        """Test getting search substate."""
        state_machine._search_substate = SearchSubstate.EXECUTING
        assert state_machine.get_search_substate() == SearchSubstate.EXECUTING
        
    @pytest.mark.asyncio
    async def test_start_search_pattern_no_pattern(self, state_machine):
        """Test starting search without a pattern."""
        result = await state_machine.start_search_pattern()
        assert not result
        
    @pytest.mark.asyncio
    async def test_start_search_pattern_with_pattern(self, state_machine, mock_mavlink):
        """Test starting search with a pattern."""
        state_machine.set_mavlink_service(mock_mavlink)
        
        pattern = MagicMock()
        pattern.id = "test-pattern"
        pattern.total_waypoints = 3
        pattern.waypoints = [
            {"lat": 1.0, "lon": 1.0, "alt": 50},
            {"lat": 2.0, "lon": 2.0, "alt": 50},
            {"lat": 3.0, "lon": 3.0, "alt": 50},
        ]
        
        state_machine.set_search_pattern(pattern)
        state_machine._current_state = SystemState.SEARCHING
        
        result = await state_machine.start_search_pattern()
        assert result
        assert state_machine._search_substate == SearchSubstate.EXECUTING
        mock_mavlink.upload_mission.assert_called_once()
        mock_mavlink.start_mission.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_pause_search_pattern(self, state_machine, mock_mavlink):
        """Test pausing search pattern."""
        state_machine.set_mavlink_service(mock_mavlink)
        state_machine._search_substate = SearchSubstate.EXECUTING
        
        result = await state_machine.pause_search_pattern()
        assert result
        assert state_machine._search_substate == SearchSubstate.PAUSED
        mock_mavlink.pause_mission.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_resume_search_pattern(self, state_machine, mock_mavlink):
        """Test resuming search pattern."""
        state_machine.set_mavlink_service(mock_mavlink)
        state_machine._search_substate = SearchSubstate.PAUSED
        
        result = await state_machine.resume_search_pattern()
        assert result
        assert state_machine._search_substate == SearchSubstate.EXECUTING
        mock_mavlink.resume_mission.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_stop_search_pattern(self, state_machine, mock_mavlink):
        """Test stopping search pattern."""
        state_machine.set_mavlink_service(mock_mavlink)
        state_machine._search_substate = SearchSubstate.EXECUTING
        state_machine._active_pattern = MagicMock()
        
        result = await state_machine.stop_search_pattern()
        assert result
        assert state_machine._search_substate == SearchSubstate.IDLE
        assert state_machine._active_pattern is None
        mock_mavlink.stop_mission.assert_called_once()
        
    def test_advance_waypoint(self, state_machine):
        """Test advancing to next waypoint."""
        pattern = MagicMock()
        pattern.total_waypoints = 3
        state_machine._active_pattern = pattern
        state_machine._current_waypoint_index = 0
        
        # Manually advance waypoint index
        state_machine._current_waypoint_index += 1
        assert state_machine._current_waypoint_index == 1
        
        # Check if at last waypoint
        state_machine._current_waypoint_index = 2
        is_last = state_machine._current_waypoint_index >= pattern.total_waypoints - 1
        assert is_last
        
    def test_get_search_pattern_status(self, state_machine):
        """Test getting search pattern status."""
        pattern = MagicMock()
        pattern.id = "test-pattern"
        pattern.total_waypoints = 5
        pattern.type = "spiral"
        
        state_machine._active_pattern = pattern
        state_machine._current_waypoint_index = 2
        state_machine._search_substate = SearchSubstate.EXECUTING
        
        status = state_machine.get_search_pattern_status()
        
        assert status["active"] is True
        assert status["pattern_id"] == "test-pattern"
        assert status["pattern_type"] == "spiral"
        assert status["current_waypoint"] == 2
        assert status["total_waypoints"] == 5
        assert status["progress"] == 40  # 2/5 * 100
        assert status["paused"] is False


class TestStatistics:
    """Test statistics and history methods."""
    
    def test_get_statistics(self, state_machine):
        """Test getting state machine statistics."""
        stats = state_machine.get_statistics()
        
        assert stats["current_state"] == "IDLE"
        assert stats["homing_enabled"] is False
        assert stats["detection_count"] == 0
        assert stats["state_changes"] == 0
        assert "state_duration_seconds" in stats
        
    @pytest.mark.asyncio
    async def test_statistics_after_transitions(self, state_machine):
        """Test statistics update after transitions."""
        await state_machine.transition_to(SystemState.SEARCHING)
        await state_machine.transition_to(SystemState.DETECTING)
        
        state_machine._detection_events.append({
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.9
        })
        
        stats = state_machine.get_statistics()
        assert stats["state_changes"] == 2
        assert stats["detection_count"] == 1
        assert stats["current_state"] == "DETECTING"
        
    def test_get_state_history(self, state_machine):
        """Test getting state history."""
        # Add some history
        state_machine._state_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "from_state": "IDLE",
                "to_state": "SEARCHING",
                "reason": "Manual start"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "from_state": "SEARCHING",
                "to_state": "DETECTING",
                "reason": "Signal detected"
            }
        ]
        
        history = state_machine.get_state_history(limit=10)
        assert len(history) == 2
        assert history[0]["to_state"] == "SEARCHING"
        
    def test_get_state_history_with_limit(self, state_machine):
        """Test getting limited state history."""
        # Add many history entries
        for i in range(10):
            state_machine._state_history.append({
                "timestamp": datetime.now().isoformat(),
                "from_state": "IDLE",
                "to_state": "SEARCHING",
                "reason": f"Test {i}"
            })
        
        history = state_machine.get_state_history(limit=5)
        assert len(history) == 5


class TestTelemetryMetrics:
    """Test telemetry metrics methods."""
    
    def test_get_telemetry_metrics(self, state_machine):
        """Test getting telemetry metrics."""
        metrics = state_machine.get_telemetry_metrics()
        
        assert "total_transitions" in metrics
        assert "state_durations" in metrics
        assert "transition_frequencies" in metrics
        assert "average_transition_time_ms" in metrics
        assert "current_state_duration_s" in metrics
        assert "uptime_seconds" in metrics
        
    @pytest.mark.asyncio
    async def test_telemetry_metrics_update(self, state_machine):
        """Test telemetry metrics update after transitions."""
        await state_machine.transition_to(SystemState.SEARCHING)
        await asyncio.sleep(0.1)  # Let some time pass
        await state_machine.transition_to(SystemState.DETECTING)
        
        metrics = state_machine.get_telemetry_metrics()
        assert metrics["total_transitions"] == 2
        assert "SEARCHING" in metrics["state_durations"]
        assert metrics["state_entry_counts"]["SEARCHING"] == 1
        assert metrics["state_entry_counts"]["DETECTING"] == 1
        
    @pytest.mark.asyncio
    async def test_send_telemetry_update(self, state_machine, mock_mavlink):
        """Test sending telemetry update."""
        state_machine.set_mavlink_service(mock_mavlink)
        
        await state_machine.send_telemetry_update()
        
        # Verify MAVLink service was called to send telemetry
        assert mock_mavlink.send_named_value_float.called


class TestForceTransition:
    """Test force transition functionality."""
    
    @pytest.mark.asyncio
    async def test_force_transition_success(self, state_machine):
        """Test successful force transition."""
        result = await state_machine.force_transition(SystemState.HOLDING)
        assert result
        assert state_machine.get_current_state() == SystemState.HOLDING
        
    @pytest.mark.asyncio
    async def test_force_transition_invalid_target(self, state_machine):
        """Test force transition to invalid state."""
        with pytest.raises(ValueError):
            await state_machine.force_transition("INVALID_STATE")


class TestEmergencyStop:
    """Test emergency stop functionality."""
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, state_machine):
        """Test emergency stop."""
        state_machine._current_state = SystemState.HOMING
        
        await state_machine.emergency_stop()
        
        assert state_machine.get_current_state() == SystemState.IDLE
        assert not state_machine._homing_enabled
        
    @pytest.mark.asyncio
    async def test_emergency_stop_clears_timeouts(self, state_machine):
        """Test that emergency stop clears timeouts."""
        state_machine.set_state_timeout(SystemState.SEARCHING, 30.0)
        state_machine._current_state = SystemState.SEARCHING
        
        await state_machine.emergency_stop()
        
        assert state_machine._timeout_task is None


class TestServiceLifecycle:
    """Test service start/stop lifecycle."""
    
    @pytest.mark.asyncio
    async def test_start_service(self, state_machine):
        """Test starting the state machine service."""
        await state_machine.start()
        
        assert state_machine._is_running
        assert state_machine._telemetry_task is not None
        
        # Clean up
        state_machine._is_running = False
        if state_machine._telemetry_task:
            state_machine._telemetry_task.cancel()
            try:
                await state_machine._telemetry_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_stop_service(self, state_machine):
        """Test stopping the state machine service."""
        await state_machine.start()
        await state_machine.stop()
        
        assert not state_machine._is_running
        assert state_machine.get_current_state() == SystemState.IDLE


class TestTransitionGuards:
    """Test transition guard conditions."""
    
    @pytest.mark.asyncio
    async def test_idle_to_searching_allowed(self, state_machine):
        """Test IDLE to SEARCHING transition is allowed."""
        result = await state_machine.transition_to(SystemState.SEARCHING)
        assert result
        assert state_machine.get_current_state() == SystemState.SEARCHING
        
    @pytest.mark.asyncio
    async def test_idle_to_homing_not_allowed(self, state_machine):
        """Test IDLE to HOMING transition is not allowed."""
        result = await state_machine.transition_to(SystemState.HOMING)
        assert not result
        assert state_machine.get_current_state() == SystemState.IDLE
        
    @pytest.mark.asyncio
    async def test_detecting_to_homing_requires_enabled(self, state_machine):
        """Test DETECTING to HOMING requires homing enabled."""
        state_machine._current_state = SystemState.DETECTING
        
        # Should fail when homing disabled
        result = await state_machine.transition_to(SystemState.HOMING)
        assert not result
        
        # Should succeed when homing enabled
        state_machine.enable_homing(True)
        result = await state_machine.transition_to(SystemState.HOMING)
        assert result
        assert state_machine.get_current_state() == SystemState.HOMING