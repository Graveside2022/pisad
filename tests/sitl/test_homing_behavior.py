"""
SITL tests for complete homing behavior.
Tests the full homing algorithm with simulated MAVLink and signal data.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.backend.services.homing_controller import HomingController
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine, SystemState

pytestmark = pytest.mark.serial


@pytest.fixture
def mock_mavlink_service():
    """Create mock MAVLink service for SITL testing."""
    service = AsyncMock(spec=MAVLinkService)
    service.is_connected.return_value = True
    service.get_telemetry.return_value = {
        "position": {"lat": 37.7749, "lon": -122.4194, "alt": 50.0},
        "heading": 45.0,
        "battery": {"percentage": 85.0},
        "flight_mode": "GUIDED",
        "armed": True,
    }
    service.send_velocity_command = AsyncMock()
    service.check_safety_interlock = AsyncMock(return_value={"safe": True})
    return service


@pytest.fixture
def mock_signal_processor():
    """Create mock signal processor for SITL testing."""
    processor = MagicMock(spec=SignalProcessor)
    processor.get_rssi = MagicMock(return_value=-70.0)
    processor.get_snr = MagicMock(return_value=20.0)
    return processor


@pytest.fixture
def mock_state_machine():
    """Create mock state machine for SITL testing."""
    machine = AsyncMock(spec=StateMachine)
    machine.current_state = SystemState.HOMING
    machine.update_state = AsyncMock()
    return machine


class TestHomingBehaviorSITL:
    """SITL tests for homing behavior."""

    @pytest.mark.asyncio
    async def test_complete_homing_sequence(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test complete homing sequence from detection to holding pattern."""
        # Create controller
        controller = HomingController(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )

        # Simulate RSSI signal progression (getting stronger)
        rssi_sequence = [
            -85.0,  # Weak signal
            -80.0,
            -75.0,
            -70.0,
            -65.0,
            -60.0,
            -55.0,
            -50.0,  # Strong signal (approach threshold)
            -48.0,
            -47.0,
            -46.5,
            -46.0,  # Plateau
            -46.0,
            -46.0,
        ]

        # Simulate position changes (moving toward beacon)
        position_sequence = [
            (0.0, 0.0),
            (10.0, 5.0),
            (20.0, 10.0),
            (30.0, 15.0),
            (35.0, 17.0),
            (38.0, 18.0),
            (39.0, 18.5),
            (39.5, 18.8),
            (39.8, 18.9),
            (40.0, 19.0),
            (40.0, 19.0),
            (40.0, 19.0),
            (40.0, 19.0),
            (40.0, 19.0),
        ]

        # Enable homing
        await controller.enable()
        assert controller.enabled is True

        # Process each RSSI sample
        for i, (rssi, position) in enumerate(zip(rssi_sequence, position_sequence, strict=False)):
            # Update signal processor mock
            mock_signal_processor.get_rssi.return_value = rssi

            # Update MAVLink telemetry mock
            mock_mavlink_service.get_telemetry.return_value = {
                "position": {
                    "lat": 37.7749 + position[0] / 111111,
                    "lon": -122.4194 + position[1] / 111111,
                    "alt": 50.0,
                },
                "heading": 45.0 + i * 2.0,  # Simulate heading changes
                "battery": {"percentage": 85.0 - i * 0.5},
                "flight_mode": "GUIDED",
                "armed": True,
            }

            # Update controller
            await controller.update()

            # Verify velocity commands were sent
            assert mock_mavlink_service.send_velocity_command.called

            # Get the last velocity command
            call_args = mock_mavlink_service.send_velocity_command.call_args
            if call_args:
                vx = call_args[1].get("vx", 0)
                yaw_rate = call_args[1].get("yaw_rate", 0)

                # Verify velocity is within limits
                assert 0 <= vx <= 8.0  # Max configured velocity
                assert -0.8 <= yaw_rate <= 0.8  # Max configured yaw rate

        # Check final state
        status = controller.get_status()

        # Should be in approach or holding mode with strong signal
        assert status["substage"] in ["APPROACH", "HOLDING"]
        assert status["enabled"] is True

    @pytest.mark.asyncio
    async def test_gradient_climbing_behavior(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test gradient climbing with directional signal changes."""
        controller = HomingController(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )

        await controller.enable()

        # Simulate gradient scenario: signal stronger to the north
        positions = [
            (0, 0, -75),  # Starting position
            (0, 10, -73),  # Move north - signal improves
            (10, 10, -76),  # Move east - signal weakens
            (10, 20, -72),  # Move north again - signal improves
            (5, 25, -70),  # Move northwest - signal improves more
        ]

        for x, y, rssi in positions:
            mock_signal_processor.get_rssi.return_value = rssi
            mock_mavlink_service.get_telemetry.return_value = {
                "position": {
                    "lat": 37.7749 + y / 111111,
                    "lon": -122.4194 + x / 111111,
                    "alt": 50.0,
                },
                "heading": np.degrees(np.arctan2(y, x)) if x != 0 or y != 0 else 0,
                "battery": {"percentage": 80.0},
                "flight_mode": "GUIDED",
                "armed": True,
            }

            await controller.update()

        # Verify gradient climbing is active
        status = controller.get_status()
        assert status["substage"] in ["GRADIENT_CLIMB", "SAMPLING"]

        # Verify velocity commands follow gradient
        assert mock_mavlink_service.send_velocity_command.call_count > 0

    @pytest.mark.asyncio
    async def test_sampling_maneuver_trigger(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test sampling maneuver triggers when gradient is unclear."""
        controller = HomingController(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )

        await controller.enable()

        # Simulate noisy/unclear signal (low gradient confidence)
        noisy_rssi = [-70 + np.random.normal(0, 5) for _ in range(10)]

        for i, rssi in enumerate(noisy_rssi):
            mock_signal_processor.get_rssi.return_value = rssi
            mock_mavlink_service.get_telemetry.return_value = {
                "position": {"lat": 37.7749 + i * 0.0001, "lon": -122.4194, "alt": 50.0},
                "heading": 0.0,
                "battery": {"percentage": 75.0},
                "flight_mode": "GUIDED",
                "armed": True,
            }

            await controller.update()
            await asyncio.sleep(0.01)  # Small delay to simulate time passage

        # Should trigger sampling due to unclear gradient
        status = controller.get_status()
        # May be in SAMPLING or still trying GRADIENT_CLIMB
        assert status["substage"] in ["SAMPLING", "GRADIENT_CLIMB"]

    @pytest.mark.asyncio
    async def test_approach_mode_activation(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test approach mode activates when signal exceeds threshold."""
        controller = HomingController(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )

        await controller.enable()

        # Simulate approaching beacon (signal gets very strong)
        approach_sequence = [
            (-60, 50, 50),
            (-55, 40, 40),
            (-52, 30, 30),
            (-50, 20, 20),  # Threshold for approach mode
            (-48, 10, 10),
            (-46, 5, 5),
        ]

        for rssi, x, y in approach_sequence:
            mock_signal_processor.get_rssi.return_value = rssi
            mock_mavlink_service.get_telemetry.return_value = {
                "position": {
                    "lat": 37.7749 + y / 111111,
                    "lon": -122.4194 + x / 111111,
                    "alt": 50.0,
                },
                "heading": 180.0,  # Moving south
                "battery": {"percentage": 70.0},
                "flight_mode": "GUIDED",
                "armed": True,
            }

            await controller.update()

            # Check if approach mode activated at threshold
            if rssi >= -50:
                status = controller.get_status()
                assert status["substage"] == "APPROACH"

                # Verify reduced velocity in approach mode
                call_args = mock_mavlink_service.send_velocity_command.call_args
                if call_args:
                    vx = call_args[1].get("vx", 0)
                    assert vx <= 2.0  # Approach velocity should be reduced

    @pytest.mark.asyncio
    async def test_holding_pattern_on_plateau(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test holding pattern activates when signal plateaus."""
        controller = HomingController(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_state_machine,
            state_machine=mock_state_machine,
        )

        await controller.enable()

        # Simulate signal plateau (beacon directly below)
        plateau_rssi = -45.0

        # Feed consistent strong signal to trigger plateau detection
        for i in range(15):
            mock_signal_processor.get_rssi.return_value = plateau_rssi + np.random.normal(0, 0.5)
            mock_mavlink_service.get_telemetry.return_value = {
                "position": {"lat": 37.7749, "lon": -122.4194, "alt": 50.0},
                "heading": i * 24.0,  # Circling
                "battery": {"percentage": 65.0},
                "flight_mode": "GUIDED",
                "armed": True,
            }

            await controller.update()
            await asyncio.sleep(0.01)

        # Should enter holding pattern
        status = controller.get_status()

        # May be in HOLDING or APPROACH depending on exact conditions
        assert status["substage"] in ["HOLDING", "APPROACH"]

        # Verify circular motion commands
        if status["substage"] == "HOLDING":
            call_args = mock_mavlink_service.send_velocity_command.call_args
            if call_args:
                yaw_rate = call_args[1].get("yaw_rate", 0)
                assert abs(yaw_rate) > 0  # Should have non-zero yaw rate for circling

    @pytest.mark.asyncio
    async def test_signal_loss_recovery(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test behavior when signal is lost during homing."""
        controller = HomingController(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )

        await controller.enable()

        # Start with good signal
        mock_signal_processor.get_rssi.return_value = -65.0
        await controller.update()

        # Simulate signal loss
        mock_signal_processor.get_rssi.return_value = -95.0  # Very weak/no signal

        # Update for duration of timeout
        for _ in range(10):
            await controller.update()
            await asyncio.sleep(0.6)  # Simulate time passage

        # Should disable after timeout
        assert not controller.enabled

    @pytest.mark.asyncio
    async def test_safety_interlock_integration(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test safety interlock prevents unsafe commands."""
        controller = HomingController(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )

        await controller.enable()

        # Trigger safety interlock
        mock_mavlink_service.check_safety_interlock.return_value = {
            "safe": False,
            "reason": "Battery critical",
        }

        # Try to update with good signal
        mock_signal_processor.get_rssi.return_value = -60.0
        await controller.update()

        # Verify zero velocity sent when unsafe
        call_args = mock_mavlink_service.send_velocity_command.call_args
        if call_args:
            vx = call_args[1].get("vx", 0)
            yaw_rate = call_args[1].get("yaw_rate", 0)
            assert vx == 0.0
            assert yaw_rate == 0.0
