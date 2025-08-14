"""Homing controller service for MAVLink integration."""

import asyncio
import logging
import math
from contextlib import suppress
from enum import Enum
from typing import Any

from backend.core.config import get_config
from backend.services.homing_algorithm import HomingAlgorithm, VelocityCommand
from backend.services.mavlink_service import MAVLinkService
from backend.services.signal_processor import SignalProcessor
from backend.services.state_machine import StateMachine

logger = logging.getLogger(__name__)


class HomingMode(str, Enum):
    """Homing algorithm selection."""

    SIMPLE = "SIMPLE"  # Basic RSSI following
    GRADIENT = "GRADIENT"  # Gradient-based algorithm


class HomingController:
    """Controller for beacon homing with MAVLink integration."""

    def __init__(
        self,
        mavlink_service: MAVLinkService,
        signal_processor: SignalProcessor,
        state_machine: StateMachine,
    ):
        """Initialize homing controller.

        Args:
            mavlink_service: MAVLink service for sending commands
            signal_processor: Signal processor for RSSI data
            state_machine: System state machine
        """
        self.mavlink = mavlink_service
        self.signal_processor = signal_processor
        self.state_machine = state_machine

        # Configuration
        config = get_config()
        homing_config = config.homing
        self.mode = HomingMode(homing_config.HOMING_ALGORITHM_MODE)
        self.signal_loss_timeout = homing_config.HOMING_SIGNAL_LOSS_TIMEOUT

        # Initialize gradient algorithm
        self.gradient_algorithm = HomingAlgorithm()

        # State tracking
        self.is_active = False
        self.last_signal_time: float | None = None
        self.update_task: asyncio.Task[None] | None = None
        self.current_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.current_heading = 0.0

        logger.info(f"Homing controller initialized with mode: {self.mode.value}")

    async def start_homing(self) -> bool:
        """Start homing toward beacon.

        Returns:
            True if homing started successfully
        """
        if self.is_active:
            logger.warning("Homing already active")
            return False

        # Check for valid signal
        latest_rssi = await self.signal_processor.get_latest_rssi()
        if latest_rssi is None or latest_rssi < -90:
            logger.error("No valid signal for homing")
            return False

        # Transition to HOMING state
        if not await self.state_machine.transition_to("HOMING"):
            logger.error("Failed to transition to HOMING state")
            return False

        self.is_active = True
        self.gradient_algorithm.reset()

        # Start update loop
        self.update_task = asyncio.create_task(self._update_loop())

        logger.info(f"Homing started with {self.mode.value} algorithm")
        return True

    async def stop_homing(self) -> bool:
        """Stop homing and return to previous state.

        Returns:
            True if homing stopped successfully
        """
        if not self.is_active:
            logger.warning("Homing not active")
            return False

        self.is_active = False

        # Cancel update task
        if self.update_task:
            self.update_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.update_task
            self.update_task = None

        # Send stop command
        await self.mavlink.send_velocity_command(0.0, 0.0, 0.0)

        # Transition to IDLE
        await self.state_machine.transition_to("IDLE")

        logger.info("Homing stopped")
        return True

    async def _update_loop(self) -> None:
        """Main update loop for homing control."""
        update_rate = 10  # Hz
        update_period = 1.0 / update_rate

        try:
            while self.is_active:
                start_time = asyncio.get_event_loop().time()

                # Update position and heading from MAVLink
                await self._update_telemetry()

                # Get latest RSSI
                rssi = await self.signal_processor.get_latest_rssi()

                if rssi is not None and rssi > -90:
                    self.last_signal_time = start_time
                    logger.debug(f"Processing RSSI: {rssi:.1f} dBm")

                    if self.mode == HomingMode.GRADIENT:
                        await self._update_gradient_homing(rssi, start_time)
                    else:
                        await self._update_simple_homing(rssi)
                else:
                    # Check for signal loss timeout
                    if self.last_signal_time:
                        time_since_signal = start_time - self.last_signal_time
                        if time_since_signal > self.signal_loss_timeout:
                            logger.warning(
                                f"Signal lost for {time_since_signal:.1f}s, stopping homing"
                            )
                            await self.stop_homing()
                            break
                        elif time_since_signal > 1.0:  # Warn after 1 second
                            logger.debug(f"No signal for {time_since_signal:.1f}s")

                # Sleep to maintain update rate
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed < update_period:
                    await asyncio.sleep(update_period - elapsed)

        except asyncio.CancelledError:
            logger.info("Homing update loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Homing update error: {e}")
            await self.stop_homing()

    async def _update_telemetry(self) -> None:
        """Update current position and heading from MAVLink."""
        try:
            telemetry = await self.mavlink.get_telemetry()
            if telemetry:
                self.current_position = {
                    "x": telemetry.get("position_x", 0.0),
                    "y": telemetry.get("position_y", 0.0),
                    "z": telemetry.get("position_z", 0.0),
                }
                self.current_heading = telemetry.get("heading", 0.0)
            else:
                logger.warning("No telemetry data available from MAVLink")
        except Exception as e:
            logger.error(f"Failed to update telemetry: {e}")

    async def _update_gradient_homing(self, rssi: float, timestamp: float) -> None:
        """Update gradient-based homing algorithm.

        Args:
            rssi: Current RSSI value in dBm
            timestamp: Current timestamp in seconds
        """
        # Add sample to gradient algorithm
        self.gradient_algorithm.add_rssi_sample(
            rssi=rssi,
            position_x=self.current_position["x"],
            position_y=self.current_position["y"],
            heading=self.current_heading,
            timestamp=timestamp,
        )

        # Calculate gradient
        gradient = self.gradient_algorithm.calculate_gradient()

        # Generate velocity command
        command = self.gradient_algorithm.generate_velocity_command(
            gradient, self.current_heading, timestamp
        )

        # Apply safety limits
        command = await self._apply_safety_limits(command)

        # Send to MAVLink
        await self.mavlink.send_velocity_command(
            vx=command.forward_velocity,
            vy=0.0,  # No lateral velocity in body frame
            yaw_rate=command.yaw_rate,
        )

        # Update state machine with substage
        await self._update_state_machine_substage()

        logger.debug(
            f"Gradient homing: vel={command.forward_velocity:.2f} m/s, "
            f"yaw={command.yaw_rate:.3f} rad/s, substage={self.gradient_algorithm.current_substage}"
        )

    async def _update_simple_homing(self, rssi: float) -> None:
        """Update simple RSSI-following homing.

        Args:
            rssi: Current RSSI value in dBm
        """
        # Simple proportional control based on RSSI
        # Stronger signal = move forward faster
        rssi_normalized = (rssi + 90) / 40  # Normalize -90 to -50 dBm to 0-1
        rssi_normalized = max(0, min(1, rssi_normalized))

        config = get_config()
        max_velocity = config.homing.HOMING_FORWARD_VELOCITY_MAX
        forward_velocity = max_velocity * rssi_normalized

        # Simple yaw control - slow rotation to search for stronger signal
        yaw_rate = 0.1 if rssi < -70 else 0.0

        # Apply safety limits
        command = VelocityCommand(forward_velocity=forward_velocity, yaw_rate=yaw_rate)
        command = await self._apply_safety_limits(command)

        # Send to MAVLink
        await self.mavlink.send_velocity_command(
            vx=command.forward_velocity, vy=0.0, yaw_rate=command.yaw_rate
        )

        logger.debug(f"Simple homing: RSSI={rssi:.1f} dBm, vel={command.forward_velocity:.2f} m/s")

    async def _apply_safety_limits(self, command: VelocityCommand) -> VelocityCommand:
        """Apply safety limits to velocity command.

        Args:
            command: Raw velocity command

        Returns:
            Safety-limited velocity command
        """
        config = get_config()
        homing_config = config.homing

        # Apply configured limits
        max_velocity = homing_config.HOMING_FORWARD_VELOCITY_MAX
        max_yaw_rate = homing_config.HOMING_YAW_RATE_MAX

        original_velocity = command.forward_velocity
        original_yaw = command.yaw_rate

        limited_velocity = max(0, min(max_velocity, command.forward_velocity))
        limited_yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, command.yaw_rate))

        # Log if limits were applied
        if limited_velocity != original_velocity or limited_yaw_rate != original_yaw:
            logger.debug(
                f"Safety limits applied: velocity {original_velocity:.2f}->{limited_velocity:.2f} m/s, "
                f"yaw {original_yaw:.3f}->{limited_yaw_rate:.3f} rad/s"
            )

        # Check safety interlock
        try:
            safety_status = await self.mavlink.check_safety_interlock()
            if not safety_status.get("safe", True):
                logger.warning(
                    f"Safety interlock triggered: {safety_status.get('reason', 'unknown')}"
                )
                limited_velocity = 0.0
                limited_yaw_rate = 0.0
        except Exception as e:
            logger.error(f"Failed to check safety interlock: {e}")
            # Fail safe - stop if can't verify safety
            limited_velocity = 0.0
            limited_yaw_rate = 0.0

        return VelocityCommand(forward_velocity=limited_velocity, yaw_rate=limited_yaw_rate)

    def get_status(self) -> dict[str, Any]:
        """Get current homing controller status.

        Returns:
            Dictionary with homing status information
        """
        status = {
            "active": self.is_active,
            "enabled": self.is_active,
            "mode": self.mode.value,
            "algorithm_mode": self.mode.value,
            "signal_loss_timeout": self.signal_loss_timeout,
            "last_rssi_time": self.last_signal_time,
            "position": self.current_position,
            "heading": self.current_heading,
            "current_heading": self.current_heading,
        }

        # Add gradient algorithm status if available
        if self.gradient_algorithm and self.mode == HomingMode.GRADIENT:
            status["algorithm_status"] = self.gradient_algorithm.get_status()
            status.update(
                {
                    "substage": self.gradient_algorithm.current_substage.value,
                    "gradient_confidence": self.gradient_algorithm.gradient_confidence,
                    "target_heading": self.gradient_algorithm.target_heading,
                    "rssi_history_size": len(self.gradient_algorithm.rssi_history),
                    "last_rssi": (
                        self.gradient_algorithm.rssi_history[-1].rssi
                        if self.gradient_algorithm.rssi_history
                        else None
                    ),
                    "velocity_command": {
                        "forward": (
                            self.gradient_algorithm.last_command.forward_velocity
                            if self.gradient_algorithm.last_command
                            else 0.0
                        ),
                        "yaw": (
                            self.gradient_algorithm.last_command.yaw_rate
                            if self.gradient_algorithm.last_command
                            else 0.0
                        ),
                    },
                }
            )

            # Add gradient direction if available
            if (
                hasattr(self.gradient_algorithm, "last_gradient")
                and self.gradient_algorithm.last_gradient
            ):
                gradient = self.gradient_algorithm.last_gradient
                # Calculate x/y from magnitude and direction
                direction_rad = math.radians(gradient.direction)
                gradient_x = gradient.magnitude * math.cos(direction_rad)
                gradient_y = gradient.magnitude * math.sin(direction_rad)
                status["gradient_direction"] = {
                    "x": gradient_x,
                    "y": gradient_y,
                    "magnitude": gradient.magnitude,
                }
        else:
            status.update(
                {
                    "substage": "IDLE",
                    "gradient_confidence": 0.0,
                    "target_heading": 0.0,
                    "rssi_history_size": 0,
                    "last_rssi": None,
                    "velocity_command": {"forward": 0.0, "yaw": 0.0},
                    "gradient_direction": None,
                }
            )

        # Add time since signal if available
        if self.last_signal_time:
            time_since_signal = asyncio.get_event_loop().time() - self.last_signal_time
            status["time_since_signal"] = time_since_signal

        return status

    async def _update_state_machine_substage(self) -> None:
        """Update state machine with current homing substage."""
        substage = self.gradient_algorithm.current_substage
        status = self.gradient_algorithm.get_status()

        # Update system state
        state_update = {
            "homing_substage": substage.value,
            "gradient_confidence": status["gradient_confidence"],
            "target_heading": status.get("gradient_direction", 0),
        }

        await self.state_machine.update_state_data(state_update)

    async def switch_mode(self, mode: str) -> bool:
        """Switch homing algorithm mode.

        Args:
            mode: "SIMPLE" or "GRADIENT"

        Returns:
            True if mode switched successfully
        """
        try:
            new_mode = HomingMode(mode.upper())
            if new_mode == self.mode:
                logger.info(f"Already in {mode} mode")
                return True

            old_mode = self.mode
            self.mode = new_mode

            # Reset gradient algorithm when switching modes
            if new_mode == HomingMode.GRADIENT:
                self.gradient_algorithm.reset()

            logger.info(f"Switched homing mode: {old_mode.value} -> {self.mode.value}")
            return True

        except ValueError:
            logger.error(
                f"Invalid homing mode: {mode}. Valid modes: {[m.value for m in HomingMode]}"
            )
            return False
