"""RSSI gradient-based homing algorithm service."""

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from backend.core.config import get_config

logger = logging.getLogger(__name__)

# Debug mode flag (updated at runtime)
_debug_mode_enabled = False


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable debug mode for verbose logging.

    Args:
        enabled: True to enable debug mode
    """
    global _debug_mode_enabled
    _debug_mode_enabled = enabled
    if enabled:
        logger.setLevel(logging.DEBUG)
        logger.info("Homing algorithm debug mode ENABLED")
    else:
        logger.setLevel(logging.INFO)
        logger.info("Homing algorithm debug mode DISABLED")


# Algorithm constants
MIN_SAMPLES_FOR_GRADIENT = 3
MIN_SPATIAL_VARIANCE = 0.1  # m²
GRADIENT_CONFIDENCE_THRESHOLD = 30  # %
HOLDING_PATTERN_YAW_RATE = 0.2  # rad/s
HOLDING_PATTERN_VELOCITY = 1.0  # m/s
SAMPLING_YAW_AMPLITUDE = 0.3  # rad/s
SAMPLING_FORWARD_VELOCITY = 2.0  # m/s
STRONG_SIGNAL_THRESHOLD = -60  # dBm
YAW_RATE_P_GAIN = 0.5  # Proportional gain for yaw control
MIN_CONFIDENT_VELOCITY = 1.0  # m/s


class HomingSubstage(str, Enum):
    """Homing algorithm substages."""

    IDLE = "IDLE"
    GRADIENT_CLIMB = "GRADIENT_CLIMB"
    SAMPLING = "SAMPLING"
    APPROACH = "APPROACH"
    HOLDING = "HOLDING"


@dataclass
class GradientVector:
    """Represents a gradient vector with magnitude and direction."""

    magnitude: float  # dB/meter
    direction: float  # degrees (0-360)
    confidence: float  # 0-100%


@dataclass
class VelocityCommand:
    """Velocity command for drone control."""

    forward_velocity: float  # m/s
    yaw_rate: float  # rad/s


@dataclass
class RSSISample:
    """RSSI sample with position and heading."""

    rssi: float  # dBm
    position_x: float  # meters
    position_y: float  # meters
    heading: float  # degrees
    timestamp: float  # seconds


class HomingAlgorithm:
    """RSSI gradient-based homing algorithm."""

    def __init__(self) -> None:
        """Initialize homing algorithm with configuration."""
        config = get_config()
        homing_config = config.homing

        # Core parameters
        self.forward_velocity_max = homing_config.HOMING_FORWARD_VELOCITY_MAX
        self.yaw_rate_max = homing_config.HOMING_YAW_RATE_MAX
        self.approach_velocity = homing_config.HOMING_APPROACH_VELOCITY

        # Gradient algorithm parameters
        self.gradient_window_size = homing_config.HOMING_GRADIENT_WINDOW_SIZE
        self.gradient_min_snr = homing_config.HOMING_GRADIENT_MIN_SNR
        self.sampling_turn_radius = homing_config.HOMING_SAMPLING_TURN_RADIUS
        self.sampling_duration = homing_config.HOMING_SAMPLING_DURATION
        self.approach_threshold = homing_config.HOMING_APPROACH_THRESHOLD
        self.plateau_variance = homing_config.HOMING_PLATEAU_VARIANCE
        self.velocity_scale_factor = homing_config.HOMING_VELOCITY_SCALE_FACTOR

        # State variables
        self.rssi_history: deque[RSSISample] = deque(maxlen=self.gradient_window_size)
        self.current_substage = HomingSubstage.IDLE
        self.sampling_start_time: float | None = None
        self.sampling_direction = 1  # 1 or -1 for S-turn direction
        self.last_gradient: GradientVector | None = None
        self.last_command: VelocityCommand | None = None
        self.gradient_confidence: float = 0.0
        self.target_heading: float = 0.0

        # Check if debug mode is enabled in config
        if (
            hasattr(config, "development")
            and hasattr(config.development, "DEV_DEBUG_MODE")
            and config.development.DEV_DEBUG_MODE
        ):
            set_debug_mode(True)

        logger.info("Homing algorithm initialized with gradient-based approach")

    def add_rssi_sample(
        self, rssi: float, position_x: float, position_y: float, heading: float, timestamp: float
    ) -> None:
        """Add new RSSI sample to history buffer.

        Args:
            rssi: Signal strength in dBm
            position_x: X position in meters (NED frame)
            position_y: Y position in meters (NED frame)
            heading: Drone heading in degrees (0-360)
            timestamp: Sample timestamp in seconds
        """
        sample = RSSISample(
            rssi=rssi,
            position_x=position_x,
            position_y=position_y,
            heading=heading,
            timestamp=timestamp,
        )
        self.rssi_history.append(sample)
        logger.debug(f"Added RSSI sample: {rssi:.1f} dBm at ({position_x:.1f}, {position_y:.1f})")

    def calculate_gradient(self) -> GradientVector | None:
        """Calculate RSSI gradient using least squares fitting.

        Returns:
            GradientVector with magnitude, direction, and confidence, or None if insufficient data
        """
        if len(self.rssi_history) < MIN_SAMPLES_FOR_GRADIENT:
            logger.debug("Insufficient samples for gradient calculation")
            return None

        # Extract data from history
        samples = list(self.rssi_history)
        positions = np.array([[s.position_x, s.position_y] for s in samples])
        rssi_values = np.array([s.rssi for s in samples])

        # Check for sufficient spatial diversity
        position_variance = np.var(positions, axis=0)
        if np.sum(position_variance) < MIN_SPATIAL_VARIANCE:
            logger.debug(
                f"Insufficient spatial diversity: total variance {np.sum(position_variance):.3f} < {MIN_SPATIAL_VARIANCE}"
            )
            return None

        # Fit plane using least squares: z = ax + by + c
        A = np.column_stack([positions, np.ones(len(samples))])
        try:
            coeffs, residuals, rank, _ = np.linalg.lstsq(A, rssi_values, rcond=None)
            # Minimum rank of 2 required for gradient in 2D space
            if rank < 2:
                logger.warning(f"Rank deficient gradient calculation: rank={rank}")
                return None

            # Extract gradient components
            grad_x, grad_y = coeffs[0], coeffs[1]

            # Calculate magnitude and direction
            magnitude = math.sqrt(grad_x**2 + grad_y**2)
            direction = math.degrees(math.atan2(grad_y, grad_x))
            direction = (direction + 360) % 360  # Normalize to 0-360

            # Calculate confidence based on fit quality (R-squared)
            if len(residuals) > 0 and len(rssi_values) > len(coeffs):
                rss = residuals[0]  # Residual sum of squares
                tss = np.sum((rssi_values - np.mean(rssi_values)) ** 2)  # Total sum of squares
                r_squared = 1 - (rss / tss) if tss > 0 else 0
                confidence = max(0, min(100, r_squared * 100))
            else:
                # Default confidence when residuals unavailable
                confidence = 50.0

            gradient = GradientVector(
                magnitude=magnitude, direction=direction, confidence=confidence
            )

            self.last_gradient = gradient
            self.gradient_confidence = confidence  # Update instance attribute
            logger.debug(
                f"Gradient: mag={magnitude:.3f} dB/m, dir={direction:.1f}°, conf={confidence:.1f}%"
            )

            return gradient

        except np.linalg.LinAlgError as e:
            logger.error(f"Gradient calculation failed: {e}")
            return None

    def compute_optimal_heading(self, gradient: GradientVector) -> float:
        """Compute optimal heading based on gradient direction.

        Args:
            gradient: Calculated gradient vector

        Returns:
            Optimal heading in degrees (0-360)
        """
        # For strongest signal increase, move in direction of positive gradient
        # Since gradient points toward increasing RSSI, follow it directly
        optimal_heading = gradient.direction

        if _debug_mode_enabled:
            logger.debug(
                f"[Heading Calc] optimal={optimal_heading:.1f}°, gradient_dir={gradient.direction:.1f}°, "
                f"gradient_conf={gradient.confidence:.1f}%"
            )
        else:
            logger.debug(f"Optimal heading: {optimal_heading:.1f}°")

        return optimal_heading

    def scale_velocity_by_gradient(self, gradient: GradientVector) -> float:
        """Scale forward velocity based on signal strength change rate.

        Args:
            gradient: Calculated gradient vector

        Returns:
            Scaled forward velocity in m/s
        """
        # Higher gradient magnitude = stronger signal change = move faster
        # Scale by velocity_scale_factor to control aggressiveness
        base_velocity = min(
            self.forward_velocity_max, gradient.magnitude * self.velocity_scale_factor
        )

        # Apply confidence scaling
        confidence_factor = gradient.confidence / 100.0
        scaled_velocity = float(base_velocity * confidence_factor)

        # Ensure minimum velocity when confident
        if gradient.confidence > 50:
            scaled_velocity = max(MIN_CONFIDENT_VELOCITY, scaled_velocity)

        logger.debug(f"Scaled velocity: {scaled_velocity:.2f} m/s")
        return scaled_velocity

    def calculate_yaw_rate(self, current_heading: float, target_heading: float) -> float:
        """Calculate yaw rate to point toward gradient direction.

        Args:
            current_heading: Current drone heading in degrees
            target_heading: Target heading in degrees

        Returns:
            Yaw rate command in rad/s
        """
        # Calculate shortest angular distance
        heading_error = target_heading - current_heading
        # Normalize to -180 to 180
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360

        # Convert to radians and apply proportional control
        heading_error_rad = math.radians(heading_error)
        yaw_rate = heading_error_rad * YAW_RATE_P_GAIN

        # Apply limits
        yaw_rate = max(-self.yaw_rate_max, min(self.yaw_rate_max, yaw_rate))

        logger.debug(f"Yaw rate: {yaw_rate:.3f} rad/s for error {heading_error:.1f}°")
        return float(yaw_rate)

    def generate_velocity_command(
        self, gradient: GradientVector | None, current_heading: float, current_time: float
    ) -> VelocityCommand:
        """Generate velocity command based on gradient and current state.

        Args:
            gradient: Calculated gradient vector (may be None)
            current_heading: Current drone heading in degrees
            current_time: Current timestamp in seconds

        Returns:
            Velocity command for drone control
        """
        # Store gradient for telemetry
        self.last_gradient = gradient

        # Update confidence and target heading for telemetry
        if gradient:
            self.gradient_confidence = gradient.confidence
            self.target_heading = self.compute_optimal_heading(gradient)
        else:
            self.gradient_confidence = 0.0
            self.target_heading = current_heading

        # Check for approach mode
        if self.rssi_history and self.rssi_history[-1].rssi > self.approach_threshold:
            if _debug_mode_enabled and self.current_substage != HomingSubstage.APPROACH:
                logger.debug(
                    f"[State Change] {self.current_substage.value} -> APPROACH, "
                    f"rssi={self.rssi_history[-1].rssi:.1f} dBm > {self.approach_threshold:.1f} dBm"
                )
            self.current_substage = HomingSubstage.APPROACH
            command = self._generate_approach_command(gradient, current_heading)
            self.last_command = command
            return command

        # Check for plateau (holding pattern)
        if self._detect_plateau():
            if _debug_mode_enabled and self.current_substage != HomingSubstage.HOLDING:
                variance = np.var([s.rssi for s in self.rssi_history]) if self.rssi_history else 0
                logger.debug(
                    f"[State Change] {self.current_substage.value} -> HOLDING, "
                    f"plateau detected, rssi_variance={variance:.2f} < {self.plateau_variance:.2f}"
                )
            self.current_substage = HomingSubstage.HOLDING
            command = self._generate_holding_command(current_time)
            self.last_command = command
            return command

        # Check gradient quality
        if gradient is None or gradient.confidence < GRADIENT_CONFIDENCE_THRESHOLD:
            if _debug_mode_enabled and self.current_substage != HomingSubstage.SAMPLING:
                logger.debug(
                    f"[State Change] {self.current_substage.value} -> SAMPLING, "
                    f"gradient_conf={gradient.confidence if gradient else 0:.1f}% < {GRADIENT_CONFIDENCE_THRESHOLD}%"
                )
            self.current_substage = HomingSubstage.SAMPLING
            command = self._generate_sampling_command(current_heading, current_time)
            self.last_command = command
            return command

        # Normal gradient climbing
        if _debug_mode_enabled and self.current_substage != HomingSubstage.GRADIENT_CLIMB:
            logger.debug(
                f"[State Change] {self.current_substage.value} -> GRADIENT_CLIMB, "
                f"gradient ready, conf={gradient.confidence:.1f}%, mag={gradient.magnitude:.3f} dB/m"
            )
        self.current_substage = HomingSubstage.GRADIENT_CLIMB
        optimal_heading = self.compute_optimal_heading(gradient)
        forward_velocity = self.scale_velocity_by_gradient(gradient)
        yaw_rate = self.calculate_yaw_rate(current_heading, optimal_heading)

        command = VelocityCommand(forward_velocity=forward_velocity, yaw_rate=yaw_rate)
        self.last_command = command

        if _debug_mode_enabled:
            logger.debug(
                f"[Command] mode=GRADIENT_CLIMB, forward={forward_velocity:.2f} m/s, "
                f"yaw={yaw_rate:.3f} rad/s, target_heading={optimal_heading:.1f}°"
            )

        return command

    def _generate_approach_command(
        self, gradient: GradientVector | None, current_heading: float
    ) -> VelocityCommand:
        """Generate reduced velocity command for close approach."""
        if gradient:
            optimal_heading = self.compute_optimal_heading(gradient)
            yaw_rate = self.calculate_yaw_rate(current_heading, optimal_heading)
        else:
            yaw_rate = 0.0

        logger.info(f"Approach mode: reduced velocity to {self.approach_velocity} m/s")
        return VelocityCommand(forward_velocity=self.approach_velocity, yaw_rate=yaw_rate)

    def _generate_holding_command(self, current_time: float) -> VelocityCommand:
        """Generate circular holding pattern command."""
        logger.info(
            f"Holding pattern: circling at {HOLDING_PATTERN_VELOCITY} m/s, yaw rate {HOLDING_PATTERN_YAW_RATE} rad/s"
        )
        return VelocityCommand(
            forward_velocity=HOLDING_PATTERN_VELOCITY, yaw_rate=HOLDING_PATTERN_YAW_RATE
        )

    def _generate_sampling_command(
        self, current_heading: float, current_time: float
    ) -> VelocityCommand:
        """Generate S-turn sampling maneuver command."""
        if self.sampling_start_time is None:
            self.sampling_start_time = current_time
            logger.info("Starting sampling maneuver")

        # Check sampling duration
        elapsed = current_time - self.sampling_start_time
        if elapsed > self.sampling_duration:
            self.sampling_start_time = None
            self.sampling_direction *= -1  # Reverse for next time
            # Try to recalculate gradient after sampling
            return VelocityCommand(forward_velocity=1.0, yaw_rate=0.0)

        # S-turn pattern
        phase = (elapsed / self.sampling_duration) * 2 * math.pi
        yaw_rate = self.sampling_direction * SAMPLING_YAW_AMPLITUDE * math.sin(phase)

        logger.debug(f"Sampling maneuver: elapsed={elapsed:.1f}s, yaw_rate={yaw_rate:.3f}")
        return VelocityCommand(forward_velocity=SAMPLING_FORWARD_VELOCITY, yaw_rate=yaw_rate)

    def get_status(self) -> dict[str, Any]:
        """Get algorithm status for telemetry.

        Returns:
            Dictionary with algorithm status information
        """
        status = {
            "substage": self.current_substage.value,
            "gradient_confidence": self.gradient_confidence,
            "target_heading": self.target_heading,
            "rssi_history_size": len(self.rssi_history),
            "last_rssi": self.rssi_history[-1].rssi if self.rssi_history else None,
            "sample_count": len(self.rssi_history),
            "gradient_magnitude": self.last_gradient.magnitude if self.last_gradient else 0,
            "gradient_direction": self.last_gradient.direction if self.last_gradient else 0,
            "debug_mode": _debug_mode_enabled,
            "last_command": {
                "forward_velocity": (
                    self.last_command.forward_velocity if self.last_command else 0.0
                ),
                "yaw_rate": self.last_command.yaw_rate if self.last_command else 0.0,
            },
        }

        # Calculate gradient x/y from magnitude and direction if gradient exists
        if self.last_gradient:
            # Convert polar (magnitude, direction) to cartesian (x, y)
            direction_rad = math.radians(self.last_gradient.direction)
            gradient_x = self.last_gradient.magnitude * math.cos(direction_rad)
            gradient_y = self.last_gradient.magnitude * math.sin(direction_rad)
            status["gradient"] = {
                "x": gradient_x,
                "y": gradient_y,
                "magnitude": self.last_gradient.magnitude,
                "confidence": self.last_gradient.confidence,
            }
        else:
            status["gradient"] = None

        if self.rssi_history:
            status["latest_rssi"] = self.rssi_history[-1].rssi

            if _debug_mode_enabled:
                # Add detailed debug information
                samples = list(self.rssi_history)
                debug_info: dict[str, Any] = {
                    "rssi_min": min(s.rssi for s in samples),
                    "rssi_max": max(s.rssi for s in samples),
                    "rssi_mean": float(np.mean([s.rssi for s in samples])),
                    "rssi_variance": float(np.var([s.rssi for s in samples])),
                    "position_spread_x": max(s.position_x for s in samples)
                    - min(s.position_x for s in samples),
                    "position_spread_y": max(s.position_y for s in samples)
                    - min(s.position_y for s in samples),
                    "time_span": (
                        samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0
                    ),
                    "sampling_active": self.sampling_start_time is not None,
                }
                status["debug_info"] = debug_info

        return status

    def _detect_plateau(self) -> bool:
        """Detect if signal has plateaued (beacon likely below).

        A plateau indicates minimal signal variation with strong signal,
        suggesting the beacon is directly below the drone.
        """
        if len(self.rssi_history) < self.gradient_window_size:
            return False

        recent_rssi = [s.rssi for s in self.rssi_history]
        variance = np.var(recent_rssi)
        mean_rssi = np.mean(recent_rssi)

        # Plateau conditions: low variance AND strong signal
        is_plateau = bool(variance < self.plateau_variance and mean_rssi > STRONG_SIGNAL_THRESHOLD)

        if is_plateau:
            logger.info(
                f"Plateau detected: variance={variance:.2f} dB², mean RSSI={mean_rssi:.1f} dBm"
            )

        return is_plateau

    def reset(self) -> None:
        """Reset algorithm state."""
        self.rssi_history.clear()
        self.current_substage = HomingSubstage.IDLE
        self.sampling_start_time = None
        self.last_gradient = None
        logger.info("Homing algorithm reset")
