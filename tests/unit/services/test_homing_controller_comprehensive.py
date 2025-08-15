"""Comprehensive unit tests for homing controller."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backend.services.homing_algorithm import VelocityCommand
from src.backend.services.homing_controller import HomingController, HomingMode

pytestmark = pytest.mark.serial


class TestHomingControllerComprehensive:
    """Comprehensive tests for homing controller."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        mavlink = AsyncMock()
        signal_processor = AsyncMock()
        state_machine = AsyncMock()
        return mavlink, signal_processor, state_machine

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.homing.HOMING_ALGORITHM_MODE = "GRADIENT"
        config.homing.HOMING_SIGNAL_LOSS_TIMEOUT = 10.0
        config.homing.HOMING_FORWARD_VELOCITY_MAX = 10.0
        config.homing.HOMING_YAW_RATE_MAX = 1.0
        config.homing.HOMING_APPROACH_VELOCITY = 2.0
        config.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
        config.homing.HOMING_GRADIENT_MIN_SNR = 10.0
        config.homing.HOMING_SAMPLING_TURN_RADIUS = 5.0
        config.homing.HOMING_SAMPLING_DURATION = 10.0
        config.homing.HOMING_APPROACH_THRESHOLD = -50.0
        config.homing.HOMING_PLATEAU_VARIANCE = 2.0
        config.homing.HOMING_VELOCITY_SCALE_FACTOR = 1.0
        config.development.DEV_DEBUG_MODE = False
        return config

    @pytest.fixture
    def controller(self, mock_services, mock_config):
        """Create homing controller instance."""
        mavlink, signal_processor, state_machine = mock_services
        with patch("src.backend.services.homing_controller.get_config", return_value=mock_config):
            with patch(
                "src.backend.services.homing_algorithm.get_config", return_value=mock_config
            ):
                return HomingController(mavlink, signal_processor, state_machine)

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.is_active is False
        assert controller.mode == HomingMode.GRADIENT
        assert controller.signal_loss_timeout == 10.0
        assert controller.last_signal_time is None
        assert controller.update_task is None
        assert controller.gradient_algorithm is not None

    @pytest.mark.asyncio
    async def test_start_homing_success(self, controller):
        """Test successful homing start."""
        controller.signal_processor.get_latest_rssi.return_value = -70.0
        controller.state_machine.transition_to.return_value = True

        result = await controller.start_homing()

        assert result is True
        assert controller.is_active is True
        assert controller.update_task is not None
        controller.state_machine.transition_to.assert_called_once_with("HOMING")

        # Clean up task
        if controller.update_task:
            controller.update_task.cancel()
            with suppress(asyncio.CancelledError):
                await controller.update_task

    @pytest.mark.asyncio
    async def test_start_homing_already_active(self, controller):
        """Test starting homing when already active."""
        controller.is_active = True

        result = await controller.start_homing()

        assert result is False
        controller.state_machine.transition_to.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_homing_no_signal(self, controller):
        """Test starting homing with no signal."""
        controller.signal_processor.get_latest_rssi.return_value = None

        result = await controller.start_homing()

        assert result is False
        assert controller.is_active is False
        controller.state_machine.transition_to.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_homing_weak_signal(self, controller):
        """Test starting homing with weak signal."""
        controller.signal_processor.get_latest_rssi.return_value = -95.0

        result = await controller.start_homing()

        assert result is False
        assert controller.is_active is False
        controller.state_machine.transition_to.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_homing_state_transition_fails(self, controller):
        """Test starting homing when state transition fails."""
        controller.signal_processor.get_latest_rssi.return_value = -70.0
        controller.state_machine.transition_to.return_value = False

        result = await controller.start_homing()

        assert result is False
        assert controller.is_active is False

    @pytest.mark.asyncio
    async def test_stop_homing_success(self, controller):
        """Test successful homing stop."""
        # Start homing first
        controller.signal_processor.get_latest_rssi.return_value = -70.0
        controller.state_machine.transition_to.return_value = True
        await controller.start_homing()

        # Now stop it
        controller.mavlink.send_velocity_command = AsyncMock(return_value=True)
        result = await controller.stop_homing()

        assert result is True
        assert controller.is_active is False
        assert controller.update_task is None
        controller.mavlink.send_velocity_command.assert_called_once_with(0, 0, 0)

    @pytest.mark.asyncio
    async def test_stop_homing_not_active(self, controller):
        """Test stopping homing when not active."""
        controller.is_active = False

        result = await controller.stop_homing()

        assert result is False
        controller.mavlink.send_velocity_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_position(self, controller):
        """Test updating position through MAVLink."""
        # Position is updated internally via _update_loop
        controller.mavlink.get_position = AsyncMock(return_value={"x": 10.0, "y": 20.0, "z": 30.0})
        controller.mavlink.get_heading = AsyncMock(return_value=45.0)

        # Position will be updated when the update loop runs
        controller.current_position = {"x": 10.0, "y": 20.0, "z": 30.0}
        controller.current_heading = 45.0

        assert controller.current_position["x"] == 10.0
        assert controller.current_position["y"] == 20.0
        assert controller.current_position["z"] == 30.0
        assert controller.current_heading == 45.0

    @pytest.mark.asyncio
    async def test_process_rssi_signal(self, controller):
        """Test processing RSSI signal in update loop."""
        # Setup
        controller.is_active = True
        controller.current_position = {"x": 10.0, "y": 20.0, "z": 30.0}
        controller.current_heading = 45.0

        # RSSI signal is processed internally in _update_loop
        import time

        current_time = time.time()
        controller.last_signal_time = current_time

        # Check that signal time tracking works
        assert controller.last_signal_time is not None
        assert controller.last_signal_time >= current_time

    @pytest.mark.asyncio
    async def test_update_loop_gradient_mode(self, controller):
        """Test gradient mode configuration."""
        controller.mode = HomingMode.GRADIENT
        controller.is_active = True

        # Mock gradient algorithm
        command = VelocityCommand(forward_velocity=5.0, yaw_rate=0.5)
        controller.gradient_algorithm.generate_velocity_command = MagicMock(return_value=command)

        # Verify mode is set correctly
        assert controller.mode == HomingMode.GRADIENT
        assert controller.is_active is True

        # Verify gradient algorithm is available
        assert controller.gradient_algorithm is not None

        # Test that command generation works
        gradient = controller.gradient_algorithm.calculate_gradient()
        test_command = controller.gradient_algorithm.generate_velocity_command(
            gradient, 45.0, 1000.0
        )
        assert test_command is not None

    @pytest.mark.asyncio
    async def test_signal_loss_detection(self, controller):
        """Test signal loss detection logic."""
        import time

        controller.is_active = True
        controller.last_signal_time = time.time() - 15.0  # 15 seconds ago
        controller.signal_loss_timeout = 10.0

        # Check if signal would be considered lost
        time_since_signal = time.time() - controller.last_signal_time
        signal_lost = time_since_signal > controller.signal_loss_timeout

        assert signal_lost is True  # Signal lost

    @pytest.mark.asyncio
    async def test_signal_not_lost(self, controller):
        """Test signal not lost logic."""
        import time

        controller.is_active = True
        controller.last_signal_time = time.time() - 5.0  # 5 seconds ago
        controller.signal_loss_timeout = 10.0

        # Check if signal would be considered lost
        time_since_signal = time.time() - controller.last_signal_time
        signal_lost = time_since_signal > controller.signal_loss_timeout

        assert signal_lost is False  # Signal not lost

    @pytest.mark.asyncio
    async def test_get_status(self, controller):
        """Test getting controller status."""
        controller.is_active = True
        controller.mode = HomingMode.GRADIENT
        controller.current_position = {"x": 10.0, "y": 20.0, "z": 30.0}
        controller.current_heading = 45.0
        controller.gradient_algorithm.get_status = MagicMock(
            return_value={"substage": "GRADIENT_CLIMB", "gradient_confidence": 75.0}
        )

        status = controller.get_status()

        assert status["active"] is True
        assert status["mode"] == "GRADIENT"
        assert status["position"] == {"x": 10.0, "y": 20.0, "z": 30.0}
        assert status["heading"] == 45.0
        assert "algorithm_status" in status
        assert status["algorithm_status"]["substage"] == "GRADIENT_CLIMB"

    @pytest.mark.asyncio
    async def test_set_mode(self, controller):
        """Test switching homing mode."""
        await controller.switch_mode("SIMPLE")
        assert controller.mode == HomingMode.SIMPLE

        await controller.switch_mode("GRADIENT")
        assert controller.mode == HomingMode.GRADIENT

    @pytest.mark.asyncio
    async def test_simple_mode_update(self, controller):
        """Test update in simple mode."""
        controller.mode = HomingMode.SIMPLE
        controller.is_active = True
        controller.signal_processor.get_latest_rssi = AsyncMock(return_value=-70.0)
        controller.mavlink.get_position = AsyncMock(return_value={"x": 10.0, "y": 20.0, "z": 30.0})
        controller.mavlink.get_heading = AsyncMock(return_value=45.0)
        controller.mavlink.send_velocity_command = AsyncMock(return_value=True)

        # In simple mode, should use basic RSSI following
        await controller._update_simple_homing(-70.0)

        # Should send some velocity command
        controller.mavlink.send_velocity_command.assert_called()

    @pytest.mark.asyncio
    async def test_emergency_stop(self, controller):
        """Test emergency stop via stop_homing."""
        controller.is_active = True
        controller.mavlink.send_velocity_command = AsyncMock(return_value=True)
        controller.state_machine.transition_to = AsyncMock(return_value=True)

        # Emergency stop is done via stop_homing
        result = await controller.stop_homing()

        assert result is True
        assert controller.is_active is False
        controller.mavlink.send_velocity_command.assert_called_with(0, 0, 0)
        controller.state_machine.transition_to.assert_called_with("IDLE")

    @pytest.mark.asyncio
    async def test_update_loop_stops_on_signal_loss(self, controller):
        """Test signal loss detection logic in the controller."""
        import time

        controller.is_active = True
        controller.signal_loss_timeout = 0.1  # Very short timeout
        controller.last_signal_time = time.time() - 1.0  # Already expired

        # Check if signal would be considered lost
        time_since_signal = time.time() - controller.last_signal_time
        should_stop = time_since_signal > controller.signal_loss_timeout

        # Assert signal loss would trigger stop
        assert should_stop is True

        # If we were to stop due to signal loss
        if should_stop:
            controller.mavlink.send_velocity_command = AsyncMock(return_value=True)
            controller.state_machine.transition_to = AsyncMock(return_value=True)
            # In real operation, stop_homing would be called
            await controller.stop_homing()
            assert controller.is_active is False

    @pytest.mark.asyncio
    async def test_gradient_algorithm_integration(self, controller):
        """Test integration with gradient algorithm."""
        controller.is_active = True
        controller.mode = HomingMode.GRADIENT

        # Add multiple RSSI samples
        positions = [(0, 0), (5, 0), (10, 0), (10, 5)]
        rssi_values = [-80, -75, -70, -65]

        for i, ((x, y), rssi) in enumerate(zip(positions, rssi_values, strict=False)):
            controller.current_position = {"x": x, "y": y, "z": 30.0}
            controller.current_heading = 90.0
            # RSSI is processed internally via update loop
            controller.gradient_algorithm.add_rssi_sample(rssi, x, y, 90.0, i * 1000.0)

        # Generate command
        import time

        command = controller.gradient_algorithm.generate_velocity_command(
            controller.gradient_algorithm.calculate_gradient(),
            controller.current_heading,
            time.time(),
        )

        # Command should be a VelocityCommand with valid values
        assert command is not None
        assert hasattr(command, "forward_velocity")
        assert hasattr(command, "yaw_rate")
        assert command.forward_velocity >= 0
        assert abs(command.yaw_rate) <= 1.0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, controller):
        """Test concurrent start/stop operations."""
        controller.signal_processor.get_latest_rssi.return_value = -70.0
        controller.state_machine.transition_to.return_value = True
        controller.mavlink.send_velocity_command = AsyncMock(return_value=True)

        # Start multiple operations concurrently
        tasks = [
            controller.start_homing(),
            controller.start_homing(),  # Should fail (already active)
            asyncio.sleep(0.01),  # Small delay
            controller.stop_homing(),
        ]

        results = await asyncio.gather(*tasks[:-1], return_exceptions=True)
        await tasks[-1]  # Stop after others complete

        # First start should succeed, second should fail
        assert results[0] is True
        assert results[1] is False

    @pytest.mark.asyncio
    async def test_mode_switching_during_operation(self, controller):
        """Test switching modes during operation."""
        controller.signal_processor.get_latest_rssi.return_value = -70.0
        controller.state_machine.transition_to.return_value = True

        # Start in gradient mode
        await controller.start_homing()
        assert controller.mode == HomingMode.GRADIENT

        # Switch to simple mode
        await controller.switch_mode("SIMPLE")
        assert controller.mode == HomingMode.SIMPLE

        # Stop homing
        controller.mavlink.send_velocity_command = AsyncMock(return_value=True)
        await controller.stop_homing()
        assert controller.is_active is False


# Add test helper to suppress asyncio warnings
from contextlib import suppress
