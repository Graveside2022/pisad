"""Doppler compensation utilities for moving platform operation.

TASK-6.1.16d - Integrate ASV Doppler compensation for moving platform operation

This module provides utilities for calculating Doppler shifts and applying
frequency corrections for SAR scenarios involving moving platforms and beacons.
"""

import math
from dataclasses import dataclass
from typing import Optional

# Speed of light constant (m/s)
SPEED_OF_LIGHT = 299_792_458


@dataclass
class PlatformVelocity:
    """Platform velocity components from MAVLink telemetry.

    Attributes:
        vx_ms: North velocity (m/s)
        vy_ms: East velocity (m/s)
        vz_ms: Down velocity (m/s)
        ground_speed_ms: Ground speed magnitude (m/s)
    """

    vx_ms: float
    vy_ms: float
    vz_ms: float
    ground_speed_ms: float


def calculate_doppler_shift(
    platform_velocity: PlatformVelocity, signal_frequency_hz: float, beacon_bearing_deg: float
) -> float:
    """Calculate Doppler shift for moving platform scenario.

    Args:
        platform_velocity: Platform velocity components
        signal_frequency_hz: Original signal frequency (Hz)
        beacon_bearing_deg: Bearing to beacon (degrees, 0=North)

    Returns:
        Doppler shift in Hz (positive = approaching, negative = receding)
    """
    # Validate inputs
    if not all(
        math.isfinite(v)
        for v in [
            platform_velocity.vx_ms,
            platform_velocity.vy_ms,
            signal_frequency_hz,
            beacon_bearing_deg,
        ]
    ):
        return 0.0

    if signal_frequency_hz <= 0:
        return 0.0

    # Convert bearing to radians
    bearing_rad = math.radians(beacon_bearing_deg)

    # Calculate radial velocity component (velocity toward beacon)
    # Positive when approaching beacon
    radial_velocity = platform_velocity.vx_ms * math.cos(
        bearing_rad
    ) + platform_velocity.vy_ms * math.sin(bearing_rad)

    # Calculate Doppler shift: f_d = f_0 * v_r / c
    doppler_shift_hz = signal_frequency_hz * radial_velocity / SPEED_OF_LIGHT

    return doppler_shift_hz


class DopplerCompensator:
    """Integrated Doppler compensation system for signal processing."""

    def __init__(self) -> None:
        """Initialize Doppler compensator."""
        pass

    def apply_compensation(
        self,
        original_frequency_hz: float,
        platform_velocity: PlatformVelocity,
        beacon_bearing_deg: float,
    ) -> float:
        """Apply Doppler compensation to correct observed frequency.

        Args:
            original_frequency_hz: Observed/nominal frequency
            platform_velocity: Current platform velocity
            beacon_bearing_deg: Bearing to beacon (degrees)

        Returns:
            Compensated frequency (Hz)
        """
        # Calculate Doppler shift
        doppler_shift_hz = calculate_doppler_shift(
            platform_velocity, original_frequency_hz, beacon_bearing_deg
        )

        # Compensate by subtracting the Doppler shift
        # This corrects the observed frequency back to the true beacon frequency
        compensated_frequency_hz = original_frequency_hz - doppler_shift_hz

        return compensated_frequency_hz
