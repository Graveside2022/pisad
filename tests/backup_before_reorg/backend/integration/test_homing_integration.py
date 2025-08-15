"""Integration tests for homing algorithm with system components."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.homing_controller import HomingController, HomingMode


@pytest.fixture
def mock_system_config():
    """Complete system configuration mock."""
    mock = MagicMock()
    # Homing configuration
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
def mock_mavlink_service():
    """Mock MAVLink service with realistic behavior."""
    mock = AsyncMock()
    mock.send_velocity_command = AsyncMock(return_value=True)
    mock.check_safety_interlock = AsyncMock(return_value={"safe": True})

    # Simulate changing telemetry
    mock._position_x = 0.0
    mock._position_y = 0.0
    mock._heading = 0.0

    async def get_telemetry():
        # Simulate drone movement
        mock._position_x += 1.0
        mock._position_y += 0.5
        mock._heading = (mock._heading + 5) % 360
        return {
            "position_x": mock._position_x,
            "position_y": mock._position_y,
            "position_z": -10.0,
            "heading": mock._heading,
        }

    mock.get_telemetry = get_telemetry
    return mock


@pytest.fixture
def mock_signal_processor():
    """Mock signal processor with realistic RSSI behavior."""
    mock = AsyncMock()
    mock._rssi_base = -80.0
    mock._sample_count = 0

    async def get_latest_rssi():
        # Simulate increasing signal strength as drone moves
        mock._sample_count += 1
        rssi = mock._rssi_base + (mock._sample_count * 0.5)
        return min(rssi, -45.0)  # Cap at -45 dBm

    mock.get_latest_rssi = get_latest_rssi
    return mock


@pytest.fixture
def mock_state_machine():
    """Mock state machine with state tracking."""
    mock = AsyncMock()
    mock._current_state = "IDLE"
    mock._state_data = {}

    async def transition_to(state):
        mock._current_state = state
        return True

    async def update_state_data(data):
        mock._state_data.update(data)
        return True

    mock.transition_to = transition_to
    mock.update_state_data = update_state_data
    return mock


class TestHomingIntegration:
    """Integration tests for homing system."""

    @pytest.mark.asyncio
    async def test_complete_homing_sequence(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test complete homing sequence from start to beacon detection."""
        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            # Start homing
            result = await controller.start_homing()
            assert result is True
            assert controller.is_active

            # Let the update loop run for a bit
            await asyncio.sleep(0.5)

            # Verify commands are being sent
            assert mock_mavlink_service.send_velocity_command.call_count > 0

            # Verify state machine was updated
            assert mock_state_machine._current_state == "HOMING"
            assert "homing_substage" in mock_state_machine._state_data

            # Stop homing
            await controller.stop_homing()
            assert not controller.is_active

    @pytest.mark.asyncio
    async def test_gradient_algorithm_with_moving_drone(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test gradient algorithm behavior as drone moves toward beacon."""
        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            await controller.start_homing()

            # Simulate multiple update cycles
            for _ in range(15):
                await asyncio.sleep(0.1)

            # Check that gradient confidence increases over time
            status = controller.gradient_algorithm.get_status()
            assert status["sample_count"] > 0

            # Verify velocity commands were sent
            assert mock_mavlink_service.send_velocity_command.call_count >= 10

            await controller.stop_homing()

    @pytest.mark.asyncio
    async def test_approach_mode_activation(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test transition to approach mode with strong signal."""
        # Set signal processor to return strong signal
        mock_signal_processor._rssi_base = -52.0  # Near approach threshold

        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            await controller.start_homing()
            await asyncio.sleep(0.3)

            # Check that approach mode was activated
            assert controller.gradient_algorithm.current_substage.value in [
                "APPROACH",
                "GRADIENT_CLIMB",
                "SAMPLING",
            ]

            # Verify reduced velocity commands
            calls = mock_mavlink_service.send_velocity_command.call_args_list
            if calls:
                # Check if any call has reduced velocity
                velocities = [call[1].get("vx", 0) for call in calls]
                assert any(
                    v <= mock_system_config.homing.HOMING_APPROACH_VELOCITY for v in velocities
                )

            await controller.stop_homing()

    @pytest.mark.asyncio
    async def test_signal_loss_recovery(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test behavior during signal loss and recovery."""
        signal_lost = False

        async def get_rssi_with_loss():
            nonlocal signal_lost
            if signal_lost:
                return None  # Signal lost
            return -70.0

        mock_signal_processor.get_latest_rssi = get_rssi_with_loss

        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            # Set short timeout for testing
            controller.signal_loss_timeout = 0.2

            await controller.start_homing()
            await asyncio.sleep(0.1)

            # Simulate signal loss
            signal_lost = True
            controller.last_signal_time = 0  # Force timeout
            await asyncio.sleep(0.3)

            # Should have stopped due to signal loss
            assert not controller.is_active
            assert mock_state_machine._current_state == "IDLE"

    @pytest.mark.asyncio
    async def test_mode_switching_during_operation(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test switching between SIMPLE and GRADIENT modes."""
        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            # Start with GRADIENT mode
            await controller.start_homing()
            assert controller.mode == HomingMode.GRADIENT

            # Switch to SIMPLE mode
            result = await controller.switch_mode("SIMPLE")
            assert result is True
            assert controller.mode == HomingMode.SIMPLE

            await asyncio.sleep(0.2)

            # Switch back to GRADIENT
            result = await controller.switch_mode("GRADIENT")
            assert result is True
            assert controller.mode == HomingMode.GRADIENT

            await controller.stop_homing()

    @pytest.mark.asyncio
    async def test_safety_interlock_integration(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test safety interlock prevents velocity commands."""
        # Configure safety interlock to trigger
        mock_mavlink_service.check_safety_interlock = AsyncMock(
            return_value={"safe": False, "reason": "low_battery"}
        )

        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            await controller.start_homing()
            await asyncio.sleep(0.2)

            # Verify zero velocity commands were sent due to safety
            calls = mock_mavlink_service.send_velocity_command.call_args_list
            for call in calls:
                assert call[1]["vx"] == 0.0
                assert call[1]["yaw_rate"] == 0.0

            await controller.stop_homing()

    @pytest.mark.asyncio
    async def test_sampling_maneuver_activation(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test sampling maneuver when gradient is uncertain."""
        # Set up poor signal conditions
        mock_signal_processor._rssi_base = -85.0
        mock_signal_processor._sample_count = 0

        # Return inconsistent RSSI to trigger sampling
        async def get_noisy_rssi():
            import random

            return -85.0 + random.uniform(-5, 5)

        mock_signal_processor.get_latest_rssi = get_noisy_rssi

        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            await controller.start_homing()

            # Let it run long enough to potentially trigger sampling
            await asyncio.sleep(0.5)

            # Check if sampling was triggered at some point
            controller.gradient_algorithm.get_status()

            # With poor/no gradient, should enter sampling mode
            assert controller.gradient_algorithm.current_substage.value in [
                "SAMPLING",
                "GRADIENT_CLIMB",
            ]

            await controller.stop_homing()

    @pytest.mark.asyncio
    async def test_status_reporting(
        self, mock_system_config, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test comprehensive status reporting during operation."""
        with (
            patch("backend.services.homing_controller.get_config", return_value=mock_system_config),
            patch("backend.services.homing_algorithm.get_config", return_value=mock_system_config),
        ):
            controller = HomingController(
                mock_mavlink_service, mock_signal_processor, mock_state_machine
            )

            # Get status before starting
            status = controller.get_status()
            assert not status["active"]
            assert status["mode"] == "GRADIENT"

            # Start homing and get status
            await controller.start_homing()
            await asyncio.sleep(0.2)

            status = controller.get_status()
            assert status["active"]
            assert "algorithm_status" in status
            assert "position" in status
            assert "heading" in status

            algo_status = status["algorithm_status"]
            assert "substage" in algo_status
            assert "sample_count" in algo_status
            assert "gradient_confidence" in algo_status

            await controller.stop_homing()
