"""Comprehensive tests for state integration service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from src.backend.services.state_integration import StateIntegration
from src.backend.services.state_machine import SystemState


@pytest.fixture
def mock_state_machine():
    """Create mock state machine."""
    mock = MagicMock()
    mock.get_current_state.return_value = SystemState.IDLE
    mock.add_state_callback = MagicMock()
    mock.set_mavlink_service = MagicMock()
    mock.set_signal_processor = MagicMock()
    mock.get_search_pattern.return_value = {"waypoints": []}
    mock.get_next_waypoint.return_value = None
    mock.update_waypoint_progress = MagicMock()
    mock.handle_detection = AsyncMock()
    mock.handle_signal_lost = AsyncMock()
    mock.transition_to = AsyncMock()
    mock.emergency_stop = AsyncMock()
    mock.start_search_pattern = AsyncMock()
    mock.stop_search_pattern = AsyncMock()
    return mock


@pytest.fixture
def mock_mavlink_service():
    """Create mock MAVLink service."""
    mock = MagicMock()
    mock.goto_waypoint = AsyncMock()
    mock.set_mode = AsyncMock()
    return mock


@pytest.fixture
def mock_signal_processor():
    """Create mock signal processor."""
    mock = MagicMock()
    mock.add_detection_callback = MagicMock()
    return mock


@pytest.fixture
def mock_homing_controller():
    """Create mock homing controller."""
    mock = MagicMock()
    mock.update_rssi = MagicMock()
    mock.start_homing = AsyncMock()
    mock.stop_homing = AsyncMock()
    return mock


@pytest.fixture
def mock_search_pattern_generator():
    """Create mock search pattern generator."""
    mock = MagicMock()
    return mock


@pytest.fixture
def state_integration(
    mock_state_machine,
    mock_mavlink_service,
    mock_signal_processor,
    mock_homing_controller,
    mock_search_pattern_generator
):
    """Create state integration instance with mocked dependencies."""
    return StateIntegration(
        state_machine=mock_state_machine,
        mavlink_service=mock_mavlink_service,
        signal_processor=mock_signal_processor,
        homing_controller=mock_homing_controller,
        search_pattern_generator=mock_search_pattern_generator
    )


class TestStateIntegrationInit:
    """Tests for StateIntegration initialization."""

    def test_init_with_all_services(self, state_integration, mock_state_machine):
        """Test initialization with all services provided."""
        assert state_integration.state_machine == mock_state_machine
        assert state_integration.mavlink_service is not None
        assert state_integration.signal_processor is not None
        assert state_integration.homing_controller is not None
        assert state_integration.search_pattern_generator is not None
        
        # Verify service connections
        mock_state_machine.set_mavlink_service.assert_called_once()
        mock_state_machine.set_signal_processor.assert_called_once()
        mock_state_machine.add_state_callback.assert_called_once()

    def test_init_minimal_services(self, mock_state_machine):
        """Test initialization with only state machine."""
        integration = StateIntegration(state_machine=mock_state_machine)
        assert integration.state_machine == mock_state_machine
        assert integration.mavlink_service is None
        assert integration.signal_processor is None
        assert integration.homing_controller is None
        assert integration.search_pattern_generator is None
        
        # Verify no service connections made
        mock_state_machine.set_mavlink_service.assert_not_called()
        mock_state_machine.set_signal_processor.assert_not_called()

    def test_signal_processor_callback_registration(
        self, mock_state_machine, mock_signal_processor
    ):
        """Test signal processor callback registration."""
        StateIntegration(
            state_machine=mock_state_machine,
            signal_processor=mock_signal_processor
        )
        mock_signal_processor.add_detection_callback.assert_called_once()


class TestStateChangeHandling:
    """Tests for state change handling."""

    @pytest.mark.asyncio
    async def test_state_change_to_searching(self, state_integration):
        """Test handling transition to SEARCHING state."""
        pattern = {"waypoints": [{"lat": 1.0, "lon": 2.0, "alt": 100}]}
        state_integration.state_machine.get_search_pattern.return_value = pattern
        
        await state_integration._on_state_change(
            SystemState.IDLE, SystemState.SEARCHING, "Starting search"
        )
        
        state_integration.state_machine.start_search_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_change_to_searching_no_pattern(self, state_integration):
        """Test SEARCHING state with no pattern loaded."""
        state_integration.state_machine.get_search_pattern.return_value = None
        
        await state_integration._on_state_change(
            SystemState.IDLE, SystemState.SEARCHING, "Starting search"
        )
        
        state_integration.state_machine.start_search_pattern.assert_not_called()

    @pytest.mark.asyncio
    async def test_state_change_to_searching_with_waypoint(self, state_integration):
        """Test SEARCHING state with waypoint navigation."""
        pattern = {"waypoints": [{"lat": 1.0, "lon": 2.0, "alt": 100}]}
        waypoint = MagicMock(latitude=1.0, longitude=2.0, altitude=100)
        state_integration.state_machine.get_search_pattern.return_value = pattern
        state_integration.state_machine.get_next_waypoint.return_value = waypoint
        
        await state_integration._on_state_change(
            SystemState.IDLE, SystemState.SEARCHING, "Starting search"
        )
        
        state_integration.mavlink_service.goto_waypoint.assert_called_once_with(
            1.0, 2.0, 100
        )

    @pytest.mark.asyncio
    async def test_state_change_to_homing(self, state_integration):
        """Test handling transition to HOMING state."""
        await state_integration._on_state_change(
            SystemState.DETECTING, SystemState.HOMING, "Signal strong"
        )
        
        state_integration.homing_controller.start_homing.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_change_to_homing_failure(self, state_integration):
        """Test homing start failure handling."""
        state_integration.homing_controller.start_homing.side_effect = Exception("Homing failed")
        
        await state_integration._on_state_change(
            SystemState.DETECTING, SystemState.HOMING, "Signal strong"
        )
        
        state_integration.state_machine.transition_to.assert_called_once_with(
            SystemState.SEARCHING, "Homing start failed"
        )

    @pytest.mark.asyncio
    async def test_state_change_to_holding(self, state_integration):
        """Test handling transition to HOLDING state."""
        await state_integration._on_state_change(
            SystemState.HOMING, SystemState.HOLDING, "Target reached"
        )
        
        state_integration.mavlink_service.set_mode.assert_called_once_with("POSHOLD")

    @pytest.mark.asyncio
    async def test_state_change_to_idle(self, state_integration):
        """Test handling transition to IDLE state."""
        await state_integration._on_state_change(
            SystemState.SEARCHING, SystemState.IDLE, "Search complete"
        )
        
        # Verify all operations stopped
        state_integration.homing_controller.stop_homing.assert_called_once()
        state_integration.state_machine.stop_search_pattern.assert_called_once()
        state_integration.mavlink_service.set_mode.assert_called_once_with("LOITER")


class TestSignalDetectionHandling:
    """Tests for signal detection handling."""

    @pytest.mark.asyncio
    async def test_signal_detection_forward(self, state_integration):
        """Test forwarding signal detection to state machine."""
        await state_integration._on_signal_detection(rssi=-75.0, confidence=0.8)
        
        state_integration.state_machine.handle_detection.assert_called_once_with(-75.0, 0.8)

    @pytest.mark.asyncio
    async def test_signal_detection_in_detecting_state(self, state_integration):
        """Test signal detection while in DETECTING state."""
        state_integration.state_machine.get_current_state.return_value = SystemState.DETECTING
        
        await state_integration._on_signal_detection(rssi=-70.0, confidence=0.9)
        
        state_integration.homing_controller.update_rssi.assert_called_once_with(-70.0)

    @pytest.mark.asyncio
    async def test_signal_detection_not_in_detecting_state(self, state_integration):
        """Test signal detection while not in DETECTING state."""
        state_integration.state_machine.get_current_state.return_value = SystemState.IDLE
        
        await state_integration._on_signal_detection(rssi=-80.0, confidence=0.5)
        
        state_integration.homing_controller.update_rssi.assert_not_called()


class TestWaypointHandling:
    """Tests for waypoint handling."""

    @pytest.mark.asyncio
    async def test_waypoint_reached_in_searching(self, state_integration):
        """Test handling waypoint reached while searching."""
        state_integration.state_machine.get_current_state.return_value = SystemState.SEARCHING
        next_waypoint = MagicMock(latitude=2.0, longitude=3.0, altitude=150)
        state_integration.state_machine.get_next_waypoint.return_value = next_waypoint
        
        await state_integration.handle_waypoint_reached(waypoint_index=0)
        
        state_integration.state_machine.update_waypoint_progress.assert_called_once_with(1)
        state_integration.mavlink_service.goto_waypoint.assert_called_once_with(
            2.0, 3.0, 150
        )

    @pytest.mark.asyncio
    async def test_waypoint_reached_pattern_complete(self, state_integration):
        """Test handling last waypoint reached."""
        state_integration.state_machine.get_current_state.return_value = SystemState.SEARCHING
        state_integration.state_machine.get_next_waypoint.return_value = None
        
        await state_integration.handle_waypoint_reached(waypoint_index=9)
        
        state_integration.state_machine.transition_to.assert_called_once_with(
            SystemState.IDLE, "Search pattern complete"
        )

    @pytest.mark.asyncio
    async def test_waypoint_reached_not_searching(self, state_integration):
        """Test waypoint reached while not searching."""
        state_integration.state_machine.get_current_state.return_value = SystemState.IDLE
        
        await state_integration.handle_waypoint_reached(waypoint_index=0)
        
        state_integration.state_machine.update_waypoint_progress.assert_not_called()


class TestSignalLostHandling:
    """Tests for signal lost handling."""

    @pytest.mark.asyncio
    async def test_signal_lost_during_detecting(self, state_integration):
        """Test handling signal lost during DETECTING state."""
        state_integration.state_machine.get_current_state.return_value = SystemState.DETECTING
        
        await state_integration.handle_signal_lost()
        
        state_integration.state_machine.handle_signal_lost.assert_called_once()
        state_integration.homing_controller.stop_homing.assert_not_called()

    @pytest.mark.asyncio
    async def test_signal_lost_during_homing(self, state_integration):
        """Test handling signal lost during HOMING state."""
        state_integration.state_machine.get_current_state.return_value = SystemState.HOMING
        
        await state_integration.handle_signal_lost()
        
        state_integration.state_machine.handle_signal_lost.assert_called_once()
        state_integration.homing_controller.stop_homing.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_lost_homing_stop_error(self, state_integration):
        """Test error handling when stopping homing after signal loss."""
        state_integration.state_machine.get_current_state.return_value = SystemState.HOMING
        state_integration.homing_controller.stop_homing.side_effect = Exception("Stop failed")
        
        await state_integration.handle_signal_lost()
        
        # Should not raise exception
        state_integration.state_machine.handle_signal_lost.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_lost_during_idle(self, state_integration):
        """Test signal lost while in IDLE state (should ignore)."""
        state_integration.state_machine.get_current_state.return_value = SystemState.IDLE
        
        await state_integration.handle_signal_lost()
        
        state_integration.state_machine.handle_signal_lost.assert_not_called()


class TestEmergencyStop:
    """Tests for emergency stop functionality."""

    @pytest.mark.asyncio
    async def test_emergency_stop_complete(self, state_integration):
        """Test complete emergency stop procedure."""
        await state_integration.emergency_stop("Test emergency")
        
        # Verify all operations stopped
        state_integration.homing_controller.stop_homing.assert_called_once()
        state_integration.state_machine.stop_search_pattern.assert_called_once()
        state_integration.mavlink_service.set_mode.assert_any_call("LOITER")
        
        # Verify state machine emergency stop
        state_integration.state_machine.emergency_stop.assert_called_once_with("Test emergency")
        
        # Verify manual mode set
        state_integration.mavlink_service.set_mode.assert_any_call("MANUAL")

    @pytest.mark.asyncio
    async def test_emergency_stop_with_errors(self, state_integration):
        """Test emergency stop with service errors."""
        state_integration.homing_controller.stop_homing.side_effect = Exception("Homing error")
        state_integration.mavlink_service.set_mode.side_effect = Exception("Mode error")
        
        # Should not raise exceptions
        await state_integration.emergency_stop("Critical failure")
        
        state_integration.state_machine.emergency_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_stop_no_mavlink(self, mock_state_machine):
        """Test emergency stop without MAVLink service."""
        integration = StateIntegration(state_machine=mock_state_machine)
        
        await integration.emergency_stop("No MAVLink")
        
        mock_state_machine.emergency_stop.assert_called_once_with("No MAVLink")


class TestServiceOperations:
    """Tests for individual service operations."""

    @pytest.mark.asyncio
    async def test_start_searching_no_generator(self, state_integration):
        """Test starting search without pattern generator."""
        state_integration.search_pattern_generator = None
        
        await state_integration._start_searching()
        
        state_integration.state_machine.start_search_pattern.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_searching_error_handling(self, state_integration):
        """Test error handling during search start."""
        pattern = {"waypoints": [{"lat": 1.0, "lon": 2.0, "alt": 100}]}
        state_integration.state_machine.get_search_pattern.return_value = pattern
        state_integration.state_machine.start_search_pattern.side_effect = Exception("Start failed")
        
        # Should not raise exception
        await state_integration._start_searching()

    @pytest.mark.asyncio
    async def test_start_homing_no_controller(self, state_integration):
        """Test starting homing without controller."""
        state_integration.homing_controller = None
        
        await state_integration._start_homing()
        
        state_integration.state_machine.transition_to.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_holding_no_mavlink(self, state_integration):
        """Test starting hold without MAVLink."""
        state_integration.mavlink_service = None
        
        await state_integration._start_holding()
        
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_start_holding_error(self, state_integration):
        """Test error handling during position hold."""
        state_integration.mavlink_service.set_mode.side_effect = Exception("Mode failed")
        
        # Should not raise exception
        await state_integration._start_holding()

    @pytest.mark.asyncio
    async def test_stop_all_operations_with_errors(self, state_integration):
        """Test stopping all operations with errors."""
        state_integration.homing_controller.stop_homing.side_effect = Exception("Stop error 1")
        state_integration.state_machine.stop_search_pattern.side_effect = Exception("Stop error 2")
        state_integration.mavlink_service.set_mode.side_effect = Exception("Mode error")
        
        # Should not raise exceptions
        await state_integration._stop_all_operations()


class TestIntegrationScenarios:
    """Tests for complete integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_search_to_homing_flow(self, state_integration):
        """Test complete flow from search to homing."""
        # Start searching
        pattern = {"waypoints": [{"lat": 1.0, "lon": 2.0, "alt": 100}]}
        waypoint = MagicMock(latitude=1.0, longitude=2.0, altitude=100)
        state_integration.state_machine.get_search_pattern.return_value = pattern
        state_integration.state_machine.get_next_waypoint.return_value = waypoint
        
        await state_integration._on_state_change(
            SystemState.IDLE, SystemState.SEARCHING, "Start search"
        )
        
        # Detect signal
        state_integration.state_machine.get_current_state.return_value = SystemState.DETECTING
        await state_integration._on_signal_detection(rssi=-60.0, confidence=0.95)
        
        # Start homing
        await state_integration._on_state_change(
            SystemState.DETECTING, SystemState.HOMING, "Strong signal"
        )
        
        # Verify complete flow
        assert state_integration.state_machine.start_search_pattern.called
        assert state_integration.mavlink_service.goto_waypoint.called
        assert state_integration.homing_controller.update_rssi.called
        assert state_integration.homing_controller.start_homing.called

    @pytest.mark.asyncio
    async def test_search_pattern_completion_flow(self, state_integration):
        """Test complete search pattern execution."""
        state_integration.state_machine.get_current_state.return_value = SystemState.SEARCHING
        
        # Process multiple waypoints
        for i in range(5):
            waypoint = MagicMock(latitude=i, longitude=i+1, altitude=100+i*10)
            state_integration.state_machine.get_next_waypoint.return_value = waypoint
            await state_integration.handle_waypoint_reached(waypoint_index=i)
        
        # Last waypoint
        state_integration.state_machine.get_next_waypoint.return_value = None
        await state_integration.handle_waypoint_reached(waypoint_index=5)
        
        # Verify pattern completion
        assert state_integration.state_machine.update_waypoint_progress.call_count == 6
        assert state_integration.mavlink_service.goto_waypoint.call_count == 5
        state_integration.state_machine.transition_to.assert_called_with(
            SystemState.IDLE, "Search pattern complete"
        )