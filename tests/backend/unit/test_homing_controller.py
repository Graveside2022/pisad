"""Unit tests for homing controller with MAVLink integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.homing_algorithm import VelocityCommand
from backend.services.homing_controller import HomingController, HomingMode


@pytest.fixture
def mock_mavlink():
    """Mock MAVLink service."""
    mock = AsyncMock()
    mock.send_velocity_command = AsyncMock(return_value=True)
    mock.get_telemetry = AsyncMock(
        return_value={
            "position_x": 0.0,
            "position_y": 0.0,
            "position_z": 0.0,
            "heading": 0.0,
        }
    )
    mock.check_safety_interlock = AsyncMock(return_value={"safe": True})
    return mock


@pytest.fixture
def mock_signal_processor():
    """Mock signal processor."""
    mock = AsyncMock()
    mock.get_latest_rssi = AsyncMock(return_value=-70.0)
    return mock


@pytest.fixture
def mock_state_machine():
    """Mock state machine."""
    mock = AsyncMock()
    mock.transition_to = AsyncMock(return_value=True)
    mock.update_state_data = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_config():
    """Mock configuration."""
    mock = MagicMock()
    mock.homing.HOMING_ALGORITHM_MODE = "GRADIENT"
    mock.homing.HOMING_SIGNAL_LOSS_TIMEOUT = 5.0
    mock.homing.HOMING_FORWARD_VELOCITY_MAX = 5.0
    mock.homing.HOMING_YAW_RATE_MAX = 0.5
    mock.homing.HOMING_APPROACH_VELOCITY = 1.0
    mock.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
    mock.homing.HOMING_APPROACH_THRESHOLD = -50.0
    mock.homing.HOMING_PLATEAU_VARIANCE = 2.0
    mock.homing.HOMING_VELOCITY_SCALE_FACTOR = 0.1
    mock.homing.HOMING_GRADIENT_MIN_SNR = 10.0
    mock.homing.HOMING_SAMPLING_TURN_RADIUS = 10.0
    mock.homing.HOMING_SAMPLING_DURATION = 5.0
    return mock


@pytest.fixture
async def homing_controller(mock_mavlink, mock_signal_processor, mock_state_machine, mock_config):
    """Create homing controller with mocked dependencies."""
    with (
        patch("backend.services.homing_controller.get_config", return_value=mock_config),
        patch("backend.services.homing_algorithm.get_config", return_value=mock_config),
    ):
        controller = HomingController(mock_mavlink, mock_signal_processor, mock_state_machine)
        yield controller
        # Cleanup
        if controller.is_active:
            await controller.stop_homing()


class TestHomingController:
    """Test suite for HomingController."""

    @pytest.mark.asyncio
    async def test_initialization(self, homing_controller):
        """Test controller initialization."""
        assert homing_controller.mode == HomingMode.GRADIENT
        assert not homing_controller.is_active
        assert homing_controller.gradient_algorithm is not None

    @pytest.mark.asyncio
    async def test_start_homing_success(
        self, homing_controller, mock_state_machine, mock_signal_processor
    ):
        """Test successful homing start."""
        result = await homing_controller.start_homing()

        assert result is True
        assert homing_controller.is_active
        assert homing_controller.update_task is not None
        mock_state_machine.transition_to.assert_called_once_with("HOMING")

    @pytest.mark.asyncio
    async def test_start_homing_already_active(self, homing_controller):
        """Test starting homing when already active."""
        await homing_controller.start_homing()
        result = await homing_controller.start_homing()

        assert result is False

    @pytest.mark.asyncio
    async def test_start_homing_no_signal(self, homing_controller, mock_signal_processor):
        """Test starting homing with no signal."""
        mock_signal_processor.get_latest_rssi.return_value = None

        result = await homing_controller.start_homing()
        assert result is False
        assert not homing_controller.is_active

    @pytest.mark.asyncio
    async def test_stop_homing(self, homing_controller, mock_mavlink, mock_state_machine):
        """Test stopping homing."""
        await homing_controller.start_homing()
        result = await homing_controller.stop_homing()

        assert result is True
        assert not homing_controller.is_active
        assert homing_controller.update_task is None
        mock_mavlink.send_velocity_command.assert_called_with(0.0, 0.0, 0.0)
        mock_state_machine.transition_to.assert_called_with("IDLE")

    @pytest.mark.asyncio
    async def test_gradient_homing_update(
        self, homing_controller, mock_mavlink, mock_signal_processor
    ):
        """Test gradient homing algorithm update."""
        # Set up telemetry
        mock_mavlink.get_telemetry.return_value = {
            "position_x": 10.0,
            "position_y": 20.0,
            "position_z": -5.0,
            "heading": 45.0,
        }

        # Manually call update to avoid timing issues
        await homing_controller._update_telemetry()
        await homing_controller._update_gradient_homing(-65.0, 1.0)

        # Verify velocity command was sent
        mock_mavlink.send_velocity_command.assert_called()
        call_args = mock_mavlink.send_velocity_command.call_args[1]
        assert "vx" in call_args
        assert "vy" in call_args
        assert "yaw_rate" in call_args

    @pytest.mark.asyncio
    async def test_simple_homing_update(self, homing_controller, mock_mavlink):
        """Test simple homing algorithm update."""
        homing_controller.mode = HomingMode.SIMPLE

        await homing_controller._update_simple_homing(-70.0)

        mock_mavlink.send_velocity_command.assert_called()
        call_args = mock_mavlink.send_velocity_command.call_args[1]
        assert call_args["vx"] > 0  # Should move forward
        assert call_args["vy"] == 0.0

    @pytest.mark.asyncio
    async def test_safety_limits(self, homing_controller):
        """Test safety limit application."""
        # Test with excessive values
        command = VelocityCommand(forward_velocity=10.0, yaw_rate=2.0)
        limited = await homing_controller._apply_safety_limits(command)

        assert limited.forward_velocity <= 5.0  # Max velocity
        assert abs(limited.yaw_rate) <= 0.5  # Max yaw rate

    @pytest.mark.asyncio
    async def test_safety_interlock(self, homing_controller, mock_mavlink):
        """Test safety interlock stops commands."""
        mock_mavlink.check_safety_interlock.return_value = {"safe": False, "reason": "test"}

        command = VelocityCommand(forward_velocity=3.0, yaw_rate=0.2)
        limited = await homing_controller._apply_safety_limits(command)

        assert limited.forward_velocity == 0.0
        assert limited.yaw_rate == 0.0

    @pytest.mark.asyncio
    async def test_signal_loss_timeout(
        self, homing_controller, mock_signal_processor, mock_mavlink
    ):
        """Test signal loss timeout stops homing."""
        await homing_controller.start_homing()
        
        # First, let it get a signal to set last_signal_time
        mock_signal_processor.get_latest_rssi.return_value = -70.0
        await asyncio.sleep(0.15)  # Let update loop process signal
        
        # Now simulate signal loss
        mock_signal_processor.get_latest_rssi.return_value = None
        
        # Fast timeout for testing
        homing_controller.signal_loss_timeout = 0.1
        
        # Wait for the update loop to detect signal loss and stop
        await asyncio.sleep(0.25)

        # Should have stopped
        assert homing_controller.is_active is False

    @pytest.mark.asyncio
    async def test_switch_mode(self, homing_controller):
        """Test switching homing modes."""
        result = await homing_controller.switch_mode("SIMPLE")
        assert result is True
        assert homing_controller.mode == HomingMode.SIMPLE

        result = await homing_controller.switch_mode("GRADIENT")
        assert result is True
        assert homing_controller.mode == HomingMode.GRADIENT

        result = await homing_controller.switch_mode("INVALID")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_status(self, homing_controller):
        """Test status reporting."""
        status = homing_controller.get_status()

        assert "active" in status
        assert "mode" in status
        assert "position" in status
        assert "heading" in status
        assert status["mode"] == "GRADIENT"
        assert not status["active"]

        # Start homing and check status
        await homing_controller.start_homing()
        status = homing_controller.get_status()

        assert status["active"]
        assert "algorithm_status" in status

    @pytest.mark.asyncio
    async def test_update_state_machine_substage(self, homing_controller, mock_state_machine):
        """Test state machine substage updates."""
        await homing_controller._update_state_machine_substage()

        mock_state_machine.update_state_data.assert_called_once()
        call_args = mock_state_machine.update_state_data.call_args[0][0]
        assert "homing_substage" in call_args
        assert "gradient_confidence" in call_args
        assert "target_heading" in call_args

    @pytest.mark.asyncio
    async def test_telemetry_update(self, homing_controller, mock_mavlink):
        """Test telemetry position and heading update."""
        mock_mavlink.get_telemetry.return_value = {
            "position_x": 100.0,
            "position_y": 200.0,
            "position_z": -50.0,
            "heading": 180.0,
        }

        await homing_controller._update_telemetry()

        assert homing_controller.current_position["x"] == 100.0
        assert homing_controller.current_position["y"] == 200.0
        assert homing_controller.current_position["z"] == -50.0
        assert homing_controller.current_heading == 180.0

    @pytest.mark.asyncio
    async def test_gradient_algorithm_integration(self, homing_controller, mock_mavlink):
        """Test integration with gradient algorithm."""
        # Add multiple samples to build gradient
        for i in range(5):
            mock_mavlink.get_telemetry.return_value = {
                "position_x": float(i * 10),
                "position_y": 0.0,
                "position_z": 0.0,
                "heading": 0.0,
            }
            await homing_controller._update_telemetry()
            await homing_controller._update_gradient_homing(-80.0 + i * 5, float(i))

        # Should have sent velocity commands
        assert mock_mavlink.send_velocity_command.call_count >= 5

    @pytest.mark.asyncio
    async def test_update_loop_cancellation(self, homing_controller):
        """Test update loop handles cancellation properly."""
        await homing_controller.start_homing()
        update_task = homing_controller.update_task

        # Stop should cancel the task
        await homing_controller.stop_homing()

        assert update_task.cancelled()
        assert homing_controller.update_task is None
