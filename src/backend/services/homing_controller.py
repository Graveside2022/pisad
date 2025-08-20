"""Homing controller service for MAVLink integration with ASV Enhancement.

TASK-6.2.2: Enhanced Homing Controller with ASV Integration
Integrates ASV professional-grade bearing calculation (±2° precision) with existing
MAVLink flight controller interface while preserving all safety mechanisms.
"""

import asyncio
import logging
import math
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any

from backend.core.config import get_config
from backend.services.homing_algorithm import HomingAlgorithm, VelocityCommand
from backend.services.mavlink_service import MAVLinkService
from backend.services.signal_processor import SignalProcessor
from backend.services.state_machine import StateMachine
from src.backend.core.exceptions import (
    DatabaseError,
    SafetyInterlockError,
)

# TASK-6.2.2 Enhanced Algorithm Integration
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVEnhancedSignalProcessor,
)


# ASV Enhanced Gradient for professional calculations
@dataclass
class ASVEnhancedGradient:
    """Enhanced gradient with ASV professional data."""

    magnitude: float
    direction: float
    confidence: float
    asv_bearing_deg: float
    asv_confidence: float
    asv_precision_deg: float
    signal_strength_dbm: float
    interference_detected: bool
    processing_method: str
    calculation_time_ms: float


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
        asv_enhanced_processor: ASVEnhancedSignalProcessor | None = None,
    ):
        """Initialize homing controller with ASV enhancement support.

        Args:
            mavlink_service: MAVLink service for sending commands
            signal_processor: Signal processor for RSSI data
            state_machine: System state machine
            asv_enhanced_processor: Optional ASV enhanced signal processor for professional algorithms
        """
        self.mavlink = mavlink_service
        self.signal_processor = signal_processor
        self.state_machine = state_machine

        # TASK-6.2.2 [24a1]: Enhanced algorithm integration interface
        self.asv_enhanced_processor = (
            asv_enhanced_processor or ASVEnhancedSignalProcessor()
        )
        self._use_asv_enhancement = (
            True  # Enable ASV professional algorithms by default
        )

        # Configuration
        config = get_config()
        homing_config = config.homing
        self.mode = HomingMode(homing_config.HOMING_ALGORITHM_MODE)
        self.signal_loss_timeout = homing_config.HOMING_SIGNAL_LOSS_TIMEOUT

        # Initialize gradient algorithm
        self.gradient_algorithm = HomingAlgorithm()

        # State tracking
        self.is_active = False
        self.homing_enabled = False  # PRD-FR14: Operator activation required
        self.last_signal_time: float | None = None
        self.update_task: asyncio.Task[None] | None = None
        self.current_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.current_heading = 0.0

        # TASK-6.2.2 Enhanced tracking
        self._last_enhanced_gradient: ASVEnhancedGradient | None = None
        self._confidence_history: list[float] = []
        self._velocity_smoothing_history: list[VelocityCommand] = []
        self._adaptive_timeouts_enabled = True

        logger.info(
            f"Homing controller initialized with mode: {self.mode.value} and ASV enhancement: {self._use_asv_enhancement}"
        )

    def enable_homing(self, confirmation: str = "") -> bool:
        """Enable homing mode per PRD-FR14 operator activation requirement.

        This method enables the homing capability but does not start homing.
        Actual homing is started separately when conditions are met.

        Args:
            confirmation: Operator confirmation token (per architecture spec)

        Returns:
            True if homing was enabled successfully
        """
        # For now, just track enabled state
        # In production, this would validate confirmation token
        self.homing_enabled = True
        logger.info(f"Homing enabled by operator with confirmation: {confirmation}")
        return True

    def disable_homing(self, reason: str = "") -> None:
        """Disable homing mode per architecture specification.

        Args:
            reason: Reason for disabling homing
        """
        self.homing_enabled = False
        logger.info(f"Homing disabled: {reason}")

    async def send_velocity_command(self, vx: float, vy: float, vz: float) -> bool:
        """Send velocity command to flight controller.

        Wrapper around MAVLink service for easier testing.

        Args:
            vx: Forward velocity (m/s)
            vy: Right velocity (m/s)
            vz: Down velocity (m/s)

        Returns:
            True if command sent successfully
        """
        return await self.mavlink.send_velocity_command(vx, vy, vz)

    async def continuous_homing_commands(self) -> None:
        """Continuously send homing velocity commands until cancelled.

        This method is used by tests to simulate continuous operation.
        Production code uses _update_loop instead.
        """
        update_rate = 10  # Hz
        update_period = 1.0 / update_rate

        try:
            while True:
                start_time = asyncio.get_event_loop().time()

                # Check if homing is still enabled and state allows commands
                if not self.homing_enabled:
                    break

                # Get flight mode from state machine
                current_mode = getattr(
                    self.state_machine, "current_flight_mode", "GUIDED"
                )
                if current_mode != "GUIDED":
                    logger.info(
                        f"Stopping velocity commands due to mode change: {current_mode}"
                    )
                    break

                # Send test velocity command
                await self.send_velocity_command(0.5, 0.0, 0.0)

                # Wait for next update
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, update_period - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Continuous homing commands cancelled")
            raise

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
        except DatabaseError as e:
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
        except DatabaseError as e:
            logger.error(f"Failed to update telemetry: {e}")

    async def _update_gradient_homing(self, rssi: float, timestamp: float) -> None:
        """Update gradient-based homing algorithm with ASV enhancement.

        TASK-6.2.2 [24a2]: Replace basic gradient calculation with ASVEnhancedSignalProcessor API.

        Args:
            rssi: Current RSSI value in dBm
            timestamp: Current timestamp in seconds
        """
        # Add sample to gradient algorithm (maintain compatibility)
        self.gradient_algorithm.add_rssi_sample(
            rssi=rssi,
            position_x=self.current_position["x"],
            position_y=self.current_position["y"],
            heading=self.current_heading,
            timestamp=timestamp,
        )

        # TASK-6.2.2: Try ASV enhanced gradient calculation first
        enhanced_gradient = await self.get_enhanced_gradient()

        if enhanced_gradient and self._use_asv_enhancement:
            # Use ASV enhanced algorithms
            logger.debug("Using ASV enhanced gradient calculation")
            command = await self.generate_enhanced_velocity_command(
                enhanced_gradient, self.current_heading
            )
        else:
            # Fallback to basic gradient calculation
            logger.debug("Using fallback gradient calculation")
            gradient = self.gradient_algorithm.calculate_gradient()
            command = self.gradient_algorithm.generate_velocity_command(
                gradient, self.current_heading, timestamp
            )

        # Apply safety limits (always applied regardless of algorithm)
        command = await self._apply_safety_limits(command)

        # Send to MAVLink
        await self.mavlink.send_velocity_command(
            vx=command.forward_velocity,
            vy=0.0,  # No lateral velocity in body frame
            yaw_rate=command.yaw_rate,
        )

        # Update state machine with substage
        await self._update_state_machine_substage()

        # Enhanced logging
        if enhanced_gradient:
            logger.debug(
                f"Enhanced gradient homing: vel={command.forward_velocity:.2f} m/s, "
                f"yaw={command.yaw_rate:.3f} rad/s, confidence={enhanced_gradient.asv_confidence:.2f}, "
                f"precision=±{enhanced_gradient.asv_precision_deg:.1f}°"
            )
        else:
            logger.debug(
                f"Basic gradient homing: vel={command.forward_velocity:.2f} m/s, "
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

        logger.debug(
            f"Simple homing: RSSI={rssi:.1f} dBm, vel={command.forward_velocity:.2f} m/s"
        )

    async def _apply_safety_limits(self, command: VelocityCommand) -> VelocityCommand:
        """Apply safety limits to velocity command.

        TASK-6.2.2 [25b1-25b4]: Safety integration with enhanced algorithms.

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

        # TASK-6.2.2: Apply confidence-based safety margins if using enhanced algorithms
        if self._use_asv_enhancement and self._last_enhanced_gradient:
            confidence = self._last_enhanced_gradient.asv_confidence

            # Lower confidence = larger safety margins
            if confidence < 0.3:
                velocity_safety_factor = 0.5  # 50% of max for very low confidence
                yaw_safety_factor = 0.6
            elif confidence < 0.6:
                velocity_safety_factor = 0.7  # 70% of max for low-medium confidence
                yaw_safety_factor = 0.8
            else:
                velocity_safety_factor = 1.0  # Full speed for high confidence
                yaw_safety_factor = 1.0

            max_velocity *= velocity_safety_factor
            max_yaw_rate *= yaw_safety_factor

            logger.debug(
                f"Enhanced safety margins: confidence={confidence:.2f}, "
                f"vel_factor={velocity_safety_factor:.2f}, yaw_factor={yaw_safety_factor:.2f}"
            )

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
        except SafetyInterlockError as e:
            logger.error(f"Failed to check safety interlock: {e}")
            # Fail safe - stop if can't verify safety
            limited_velocity = 0.0
            limited_yaw_rate = 0.0

        return VelocityCommand(
            forward_velocity=limited_velocity, yaw_rate=limited_yaw_rate
        )

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

    # TASK-6.2.2 Enhanced Algorithm Integration Methods

    async def get_enhanced_gradient(self) -> ASVEnhancedGradient | None:
        """Get enhanced gradient using ASV professional algorithms.

        TASK-6.2.2 [24a1]: Enhanced algorithm integration interface.

        Returns:
            ASVEnhancedGradient with professional-grade bearing data or None if unavailable
        """
        if not self.asv_enhanced_processor or not self._use_asv_enhancement:
            return None

        try:
            # Get current RSSI and position data
            latest_rssi = await self.signal_processor.get_latest_rssi()
            if latest_rssi is None:
                return None

            # Create mock RSSI history for ASV calculation
            # In real implementation, this would come from actual RSSI history
            rssi_samples = [
                {
                    "rssi": latest_rssi,
                    "position_x": self.current_position["x"],
                    "position_y": self.current_position["y"],
                    "heading": self.current_heading,
                    "timestamp": time.time(),
                }
            ]

            # Use ASV enhanced processor to get professional bearing with RSSI samples
            enhanced_gradient = await self.asv_enhanced_processor.get_enhanced_gradient(
                rssi_samples=rssi_samples
            )
            if enhanced_gradient:
                # Store for tracking
                self._last_enhanced_gradient = enhanced_gradient
                self._confidence_history.append(enhanced_gradient.asv_confidence)

                # Limit history size
                if len(self._confidence_history) > 50:
                    self._confidence_history.pop(0)

                logger.debug(
                    f"Enhanced gradient: bearing={enhanced_gradient.asv_bearing_deg:.1f}°, "
                    f"confidence={enhanced_gradient.asv_confidence:.2f}, "
                    f"precision=±{enhanced_gradient.asv_precision_deg:.1f}°"
                )

                return enhanced_gradient

        except Exception as e:
            logger.error(f"Enhanced gradient calculation failed: {e}")

        return None

    async def generate_enhanced_velocity_command(
        self, enhanced_gradient: ASVEnhancedGradient, current_heading: float
    ) -> VelocityCommand:
        """Generate velocity command using ASV enhanced algorithms.

        TASK-6.2.2 [24a3]: Enhanced bearing precision integration.
        TASK-6.2.2 [24d1-24d4]: Enhanced command generation with latency validation.

        Args:
            enhanced_gradient: ASV enhanced gradient with professional data
            current_heading: Current drone heading in degrees

        Returns:
            VelocityCommand with enhanced precision and confidence weighting
        """
        start_time = time.time()

        # Calculate optimal heading using enhanced precision
        optimal_heading = enhanced_gradient.asv_bearing_deg

        # Calculate confidence-scaled velocity
        confidence_scaled_velocity = await self.calculate_confidence_scaled_velocity(
            enhanced_gradient
        )

        # Calculate precision-aware yaw rate
        heading_error = optimal_heading - current_heading
        # Normalize to -180 to 180
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360

        # Enhanced precision allows for more aggressive yaw rates
        precision_factor = max(
            0.1, 2.0 / enhanced_gradient.asv_precision_deg
        )  # Better precision = higher factor
        yaw_rate = math.radians(heading_error) * precision_factor * 0.5

        # Apply safety limits
        config = get_config()
        max_yaw_rate = config.homing.HOMING_YAW_RATE_MAX
        yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, yaw_rate))

        # Create velocity command
        command = VelocityCommand(
            forward_velocity=confidence_scaled_velocity, yaw_rate=yaw_rate
        )

        # Apply confidence-weighted smoothing [24d3]
        command = await self._apply_confidence_weighted_smoothing(
            command, enhanced_gradient
        )

        # Apply safety limits
        command = await self._apply_safety_limits(command)

        # Validate latency requirement [24d4]
        processing_time_ms = (time.time() - start_time) * 1000
        if processing_time_ms > 100.0:
            logger.warning(
                f"Enhanced command generation took {processing_time_ms:.2f}ms, exceeds 100ms limit"
            )

        # Add enhanced metadata for tracking
        command.confidence_weighted_smoothing = True

        # Store in smoothing history
        self._velocity_smoothing_history.append(command)
        if len(self._velocity_smoothing_history) > 10:
            self._velocity_smoothing_history.pop(0)

        logger.debug(
            f"Enhanced velocity command: forward={command.forward_velocity:.2f}m/s, "
            f"yaw={command.yaw_rate:.3f}rad/s, processing_time={processing_time_ms:.1f}ms"
        )

        return command

    async def calculate_confidence_scaled_velocity(
        self, enhanced_gradient: ASVEnhancedGradient
    ) -> float:
        """Calculate velocity scaled by ASV confidence metrics.

        TASK-6.2.2 [24b1-24b4]: Confidence-based velocity scaling.

        Args:
            enhanced_gradient: Enhanced gradient with confidence data

        Returns:
            Confidence-scaled forward velocity
        """
        config = get_config()
        max_velocity = config.homing.HOMING_FORWARD_VELOCITY_MAX

        # Base velocity from gradient magnitude
        base_velocity = min(max_velocity, enhanced_gradient.magnitude * 2.0)

        # Confidence-based scaling
        confidence = enhanced_gradient.asv_confidence
        if confidence >= 0.80:
            # High confidence: near full speed
            velocity_multiplier = 0.9 + (confidence - 0.80) * 0.5  # 0.9 to 1.0
        elif confidence >= 0.50:
            # Medium confidence: scaled speed
            velocity_multiplier = 0.5 + (confidence - 0.50) * 1.33  # 0.5 to 0.9
        else:
            # Low confidence: cautious speed
            velocity_multiplier = 0.2 + confidence * 0.6  # 0.2 to 0.5

        # Factor in signal strength for additional scaling
        signal_strength_factor = max(
            0.3, min(1.0, (enhanced_gradient.signal_strength_dbm + 100) / 50)
        )

        # Apply interference penalty
        interference_penalty = 0.7 if enhanced_gradient.interference_detected else 1.0

        scaled_velocity = (
            base_velocity
            * velocity_multiplier
            * signal_strength_factor
            * interference_penalty
        )

        # Ensure minimum velocity for all signals
        scaled_velocity = max(scaled_velocity, 0.5)  # Minimum 0.5 m/s for any movement

        # Higher minimum for confident signals
        if confidence > 0.60:
            scaled_velocity = max(
                scaled_velocity, 1.0
            )  # Minimum 1 m/s for confident signals

        logger.debug(
            f"Velocity scaling: base={base_velocity:.2f}, confidence={confidence:.2f}, "
            f"multiplier={velocity_multiplier:.2f}, final={scaled_velocity:.2f}"
        )

        return scaled_velocity

    async def make_confidence_based_decision(
        self, enhanced_gradient: ASVEnhancedGradient
    ) -> dict[str, Any]:
        """Make homing decisions based on ASV confidence metrics.

        TASK-6.2.2 [24a4]: Enhanced confidence metrics integration.

        Args:
            enhanced_gradient: Enhanced gradient with confidence data

        Returns:
            Decision dictionary with strategy and confidence level
        """
        confidence = enhanced_gradient.asv_confidence

        if confidence >= 0.80:
            strategy = "aggressive_homing"
            confidence_level = "high"
        elif confidence >= 0.50:
            strategy = "standard_homing"
            confidence_level = "medium"
        elif confidence > 0.30:  # Changed from >= to >
            strategy = "cautious_homing"
            confidence_level = "low"
        else:  # 30% and below
            strategy = "cautious_sampling"
            confidence_level = "low"

        # Factor in interference detection
        if enhanced_gradient.interference_detected and strategy == "aggressive_homing":
            strategy = "cautious_homing"

        # Factor in precision degradation
        if (
            enhanced_gradient.asv_precision_deg > 5.0
            and strategy == "aggressive_homing"
        ):
            strategy = "standard_homing"

        decision = {
            "strategy": strategy,
            "confidence_level": confidence_level,
            "confidence_value": confidence,
            "precision_deg": enhanced_gradient.asv_precision_deg,
            "interference_detected": enhanced_gradient.interference_detected,
            "recommended_timeout": await self.calculate_adaptive_timeout(
                confidence, enhanced_gradient.signal_strength_dbm
            ),
        }

        logger.debug(f"Confidence-based decision: {decision}")
        return decision

    async def calculate_adaptive_timeout(
        self, confidence: float, signal_strength_dbm: float
    ) -> float:
        """Calculate adaptive timeout based on signal confidence.

        TASK-6.2.2 [24c1-24c4]: Adaptive timeout configuration.

        Args:
            confidence: Signal confidence (0.0-1.0)
            signal_strength_dbm: Signal strength in dBm

        Returns:
            Adaptive timeout in seconds
        """
        if not self._adaptive_timeouts_enabled:
            return self.signal_loss_timeout  # Default timeout

        # Base timeout from configuration
        base_timeout = self.signal_loss_timeout

        # Confidence-based scaling
        if confidence >= 0.80:
            # High confidence: extend timeout (more patience)
            timeout_multiplier = 1.5 + (confidence - 0.80) * 1.0  # 1.5 to 2.0
        elif confidence >= 0.50:
            # Medium confidence: standard timeout
            timeout_multiplier = 1.0 + (confidence - 0.50) * 1.67  # 1.0 to 1.5
        else:
            # Low confidence: shorter timeout (quick fallback)
            timeout_multiplier = 0.5 + confidence * 1.0  # 0.5 to 1.0

        # Signal strength factor
        # Strong signals (-40 to -60 dBm) get longer timeouts
        # Weak signals (-80 to -100 dBm) get shorter timeouts
        if signal_strength_dbm > -60:
            strength_multiplier = 1.2
        elif signal_strength_dbm > -80:
            strength_multiplier = 1.0
        else:
            strength_multiplier = 0.8

        adaptive_timeout = base_timeout * timeout_multiplier * strength_multiplier

        # Apply reasonable bounds
        adaptive_timeout = max(5.0, min(60.0, adaptive_timeout))

        logger.debug(
            f"Adaptive timeout: base={base_timeout}s, confidence={confidence:.2f}, "
            f"signal={signal_strength_dbm:.1f}dBm, final={adaptive_timeout:.1f}s"
        )

        return adaptive_timeout

    async def _apply_confidence_weighted_smoothing(
        self, command: VelocityCommand, enhanced_gradient: ASVEnhancedGradient
    ) -> VelocityCommand:
        """Apply confidence-weighted command smoothing.

        TASK-6.2.2 [24d3]: Confidence-weighted smoothing to reduce oscillation.

        Args:
            command: Raw velocity command
            enhanced_gradient: Enhanced gradient with confidence data

        Returns:
            Smoothed velocity command
        """
        if not self._velocity_smoothing_history:
            return command  # No history for smoothing

        # Get recent commands for smoothing
        recent_commands = self._velocity_smoothing_history[-3:]  # Last 3 commands
        if not recent_commands:
            return command

        # Confidence-based smoothing factor
        confidence = enhanced_gradient.asv_confidence
        if confidence >= 0.80:
            smoothing_factor = 0.1  # Minimal smoothing for high confidence
        elif confidence >= 0.50:
            smoothing_factor = 0.3  # Moderate smoothing
        else:
            smoothing_factor = 0.6  # Heavy smoothing for low confidence

        # Calculate weighted average with recent commands
        total_weight = 1.0  # Current command weight
        weighted_forward = command.forward_velocity * 1.0
        weighted_yaw = command.yaw_rate * 1.0

        for i, prev_command in enumerate(recent_commands):
            weight = smoothing_factor * (0.5**i)  # Exponential decay
            total_weight += weight
            weighted_forward += prev_command.forward_velocity * weight
            weighted_yaw += prev_command.yaw_rate * weight

        smoothed_command = VelocityCommand(
            forward_velocity=weighted_forward / total_weight,
            yaw_rate=weighted_yaw / total_weight,
        )

        logger.debug(
            f"Command smoothing: original=({command.forward_velocity:.2f}, {command.yaw_rate:.3f}), "
            f"smoothed=({smoothed_command.forward_velocity:.2f}, {smoothed_command.yaw_rate:.3f}), "
            f"factor={smoothing_factor:.2f}"
        )

        return smoothed_command

    async def emergency_stop(self) -> bool:
        """Emergency stop with enhanced algorithm cleanup.

        TASK-6.2.2 [25d1-25d4]: Emergency stop integration with enhanced algorithms.

        Returns:
            True if emergency stop was successful
        """
        start_time = time.time()

        try:
            # Stop homing immediately
            success = await self.stop_homing()

            # Clean up enhanced algorithm state
            await self.cleanup_enhanced_algorithm_state()

            # Send emergency stop to MAVLink
            await self.mavlink.send_velocity_command(0.0, 0.0, 0.0)

            # Verify timing requirement
            stop_time_ms = (time.time() - start_time) * 1000
            if stop_time_ms > 500.0:
                logger.error(
                    f"Emergency stop took {stop_time_ms:.2f}ms, exceeds 500ms requirement"
                )
            else:
                logger.info(f"Emergency stop completed in {stop_time_ms:.2f}ms")

            return success

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

    async def cleanup_enhanced_algorithm_state(self) -> None:
        """Clean up enhanced algorithm state during emergency stop.

        TASK-6.2.2 [25d4]: Enhanced algorithm state cleanup.
        """
        try:
            # Clear enhanced algorithm data
            self._last_enhanced_gradient = None
            self._confidence_history.clear()
            self._velocity_smoothing_history.clear()

            # Reset ASV processor if available
            if self.asv_enhanced_processor:
                # Reset any internal state in ASV processor
                # Implementation would depend on ASV processor interface
                pass

            logger.debug("Enhanced algorithm state cleaned up")

        except Exception as e:
            logger.error(f"Enhanced algorithm cleanup failed: {e}")

    async def verify_enhanced_algorithm_cleanup(self) -> bool:
        """Verify enhanced algorithm state was properly cleaned up.

        Returns:
            True if cleanup was successful
        """
        return (
            self._last_enhanced_gradient is None
            and len(self._confidence_history) == 0
            and len(self._velocity_smoothing_history) == 0
        )

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
