"""Unit tests for Doppler compensation functionality.

TASK-6.1.16d - Integrate ASV Doppler compensation for moving platform operation

This test suite validates the Doppler compensation functionality including:
- Platform velocity extraction from MAVLink GLOBAL_POSITION_INT messages
- Doppler shift calculation algorithms
- Frequency correction for moving platform scenarios
- Integration with ASV enhanced signal processor
"""

import math
from unittest.mock import Mock

import pytest

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.utils.doppler_compensation import (
    DopplerCompensator,
    PlatformVelocity,
    calculate_doppler_shift,
)


class TestPlatformVelocityExtraction:
    """Test suite for platform velocity extraction from MAVLink messages."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        return MAVLinkService(device_path="tcp:127.0.0.1:5760")  # SITL connection

    @pytest.mark.asyncio
    async def test_velocity_extraction_from_global_position_int(self, mavlink_service):
        """Test velocity extraction from GLOBAL_POSITION_INT message.

        This test verifies that platform velocity is correctly extracted from
        MAVLink GLOBAL_POSITION_INT messages and made available for Doppler calculations.
        """
        # Arrange - Create mock GLOBAL_POSITION_INT message with velocity data
        mock_msg = Mock()
        mock_msg.vx = 500  # cm/s North velocity (5 m/s)
        mock_msg.vy = -300  # cm/s East velocity (-3 m/s)
        mock_msg.vz = 100  # cm/s Down velocity (1 m/s)
        mock_msg.lat = 37.7749 * 1e7  # San Francisco lat in 1e7 degrees
        mock_msg.lon = -122.4194 * 1e7  # San Francisco lon in 1e7 degrees
        mock_msg.alt = 100 * 1000  # 100m altitude in mm

        # Act - Process the message
        mavlink_service._process_global_position(mock_msg)

        # Assert - Verify velocity is extracted and available
        velocity = mavlink_service.get_platform_velocity()
        assert velocity is not None
        assert isinstance(velocity, PlatformVelocity)
        assert abs(velocity.vx_ms - 5.0) < 0.01  # 500 cm/s = 5.0 m/s
        assert abs(velocity.vy_ms - (-3.0)) < 0.01  # -300 cm/s = -3.0 m/s
        assert abs(velocity.vz_ms - 1.0) < 0.01  # 100 cm/s = 1.0 m/s

        # Verify ground speed calculation
        expected_ground_speed = math.sqrt(5.0**2 + 3.0**2)  # ~5.83 m/s
        assert abs(velocity.ground_speed_ms - expected_ground_speed) < 0.01

    @pytest.mark.asyncio
    async def test_velocity_unavailable_when_no_messages(self, mavlink_service):
        """Test that velocity is unavailable when no GLOBAL_POSITION_INT received."""
        # Act - Get velocity without processing any messages
        velocity = mavlink_service.get_platform_velocity()

        # Assert - Should return None or zero velocity
        assert velocity is None or (
            velocity.vx_ms == 0.0 and velocity.vy_ms == 0.0 and velocity.vz_ms == 0.0
        )


class TestDopplerShiftCalculation:
    """Test suite for Doppler shift calculation algorithms."""

    def test_doppler_shift_approaching_beacon(self):
        """Test Doppler shift calculation for approaching beacon scenario."""
        # Arrange - Platform moving toward beacon at 10 m/s
        platform_velocity = PlatformVelocity(vx_ms=10.0, vy_ms=0.0, vz_ms=0.0, ground_speed_ms=10.0)
        signal_frequency_hz = 406_000_000  # 406 MHz emergency beacon
        beacon_bearing_deg = 0.0  # Beacon directly ahead (North)

        # Act - Calculate Doppler shift
        doppler_shift_hz = calculate_doppler_shift(
            platform_velocity, signal_frequency_hz, beacon_bearing_deg
        )

        # Assert - Should be positive shift (higher frequency) when approaching
        # Expected: f_d = f_0 * (v * cos(0°)) / c = 406MHz * 10 / 299792458 ≈ 13.5 Hz
        expected_shift = signal_frequency_hz * 10.0 / 299_792_458  # Speed of light
        assert abs(doppler_shift_hz - expected_shift) < 0.1
        assert doppler_shift_hz > 0  # Positive shift when approaching

    def test_doppler_shift_receding_beacon(self):
        """Test Doppler shift calculation for receding beacon scenario."""
        # Arrange - Platform moving away from beacon at 10 m/s
        platform_velocity = PlatformVelocity(
            vx_ms=-10.0, vy_ms=0.0, vz_ms=0.0, ground_speed_ms=10.0
        )
        signal_frequency_hz = 406_000_000  # 406 MHz emergency beacon
        beacon_bearing_deg = 0.0  # Beacon behind (North, but moving South)

        # Act - Calculate Doppler shift
        doppler_shift_hz = calculate_doppler_shift(
            platform_velocity, signal_frequency_hz, beacon_bearing_deg
        )

        # Assert - Should be negative shift (lower frequency) when receding
        expected_shift = signal_frequency_hz * (-10.0) / 299_792_458
        assert abs(doppler_shift_hz - expected_shift) < 0.1
        assert doppler_shift_hz < 0  # Negative shift when receding

    def test_doppler_shift_perpendicular_motion(self):
        """Test Doppler shift calculation for perpendicular motion (no radial component)."""
        # Arrange - Platform moving perpendicular to beacon direction
        platform_velocity = PlatformVelocity(vx_ms=0.0, vy_ms=20.0, vz_ms=0.0, ground_speed_ms=20.0)
        signal_frequency_hz = 406_000_000  # 406 MHz emergency beacon
        beacon_bearing_deg = 0.0  # Beacon to North, platform moving East

        # Act - Calculate Doppler shift
        doppler_shift_hz = calculate_doppler_shift(
            platform_velocity, signal_frequency_hz, beacon_bearing_deg
        )

        # Assert - Should be near zero shift for perpendicular motion
        assert abs(doppler_shift_hz) < 1.0  # Within 1 Hz of zero


class TestDopplerCompensator:
    """Test suite for integrated Doppler compensation system."""

    @pytest.fixture
    def doppler_compensator(self):
        """Create Doppler compensator for testing."""
        return DopplerCompensator()

    def test_frequency_correction_integration(self, doppler_compensator):
        """Test frequency correction integration with signal processing."""
        # Arrange
        original_frequency_hz = 406_000_000
        platform_velocity = PlatformVelocity(vx_ms=15.0, vy_ms=0.0, vz_ms=0.0, ground_speed_ms=15.0)
        beacon_bearing_deg = 0.0

        # Act - Apply Doppler compensation
        corrected_frequency_hz = doppler_compensator.apply_compensation(
            original_frequency_hz, platform_velocity, beacon_bearing_deg
        )

        # Assert - Corrected frequency should differ from original
        assert corrected_frequency_hz != original_frequency_hz
        # For approaching beacon, corrected frequency should be lower to compensate for positive Doppler
        assert corrected_frequency_hz < original_frequency_hz

        # Verify correction magnitude is reasonable for SAR velocities
        frequency_difference = abs(corrected_frequency_hz - original_frequency_hz)
        assert 1.0 <= frequency_difference <= 100.0  # 1-100 Hz range for typical SAR speeds
