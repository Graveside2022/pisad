"""Unit tests for Doppler compensation integration with homing algorithm.

TASK-6.1.16d - Test Doppler compensation integration with existing homing algorithm
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.backend.services.homing_algorithm import HomingAlgorithm, GradientVector
from src.backend.utils.doppler_compensation import PlatformVelocity


class TestHomingDopplerIntegration:
    """Test suite for Doppler compensation integration with homing algorithm."""

    @pytest.fixture
    def homing_algorithm(self):
        """Create homing algorithm instance for testing."""
        return HomingAlgorithm()

    @pytest.fixture
    def platform_velocity(self):
        """Create test platform velocity."""
        return PlatformVelocity(vx_ms=10.0, vy_ms=5.0, vz_ms=0.0, ground_speed_ms=11.18)

    def test_platform_velocity_setting(self, homing_algorithm, platform_velocity):
        """Test setting platform velocity for Doppler compensation."""
        # Act - Set platform velocity
        homing_algorithm.set_platform_velocity(platform_velocity)
        
        # Assert - Velocity is stored correctly
        stored_velocity = homing_algorithm._current_platform_velocity
        assert stored_velocity is not None
        assert stored_velocity.vx_ms == 10.0
        assert stored_velocity.vy_ms == 5.0
        assert stored_velocity.ground_speed_ms == 11.18

    def test_signal_frequency_setting(self, homing_algorithm):
        """Test setting signal frequency for Doppler compensation."""
        # Arrange
        test_frequency = 121_500_000  # 121.5 MHz aviation frequency
        
        # Act
        homing_algorithm.set_signal_frequency(test_frequency)
        
        # Assert
        assert homing_algorithm._signal_frequency_hz == test_frequency

    def test_doppler_compensation_applied(self, homing_algorithm, platform_velocity):
        """Test that Doppler compensation is applied to gradient calculations."""
        # Arrange - Set up velocity and frequency
        homing_algorithm.set_platform_velocity(platform_velocity)
        homing_algorithm.set_signal_frequency(406_000_000)
        
        # Create test gradient
        original_gradient = GradientVector(
            magnitude=0.5,
            direction=45.0,  # NE direction
            confidence=0.8
        )
        
        # Act - Apply Doppler compensation
        compensated_gradient = homing_algorithm._apply_doppler_compensation(original_gradient)
        
        # Assert - Compensation was applied
        assert compensated_gradient is not None
        assert isinstance(compensated_gradient, GradientVector)
        # Magnitude should be adjusted based on Doppler effect
        assert compensated_gradient.magnitude != original_gradient.magnitude
        # Direction should remain the same
        assert compensated_gradient.direction == original_gradient.direction
        # Confidence should be slightly reduced
        assert compensated_gradient.confidence <= original_gradient.confidence

    def test_doppler_compensation_disabled_no_velocity(self, homing_algorithm):
        """Test that Doppler compensation is skipped when no velocity available."""
        # Arrange - No platform velocity set
        original_gradient = GradientVector(
            magnitude=0.5,
            direction=45.0,
            confidence=0.8
        )
        
        # Act - Apply compensation without velocity
        compensated_gradient = homing_algorithm._apply_doppler_compensation(original_gradient)
        
        # Assert - Original gradient unchanged
        assert compensated_gradient == original_gradient
        assert compensated_gradient.magnitude == original_gradient.magnitude
        assert compensated_gradient.direction == original_gradient.direction
        assert compensated_gradient.confidence == original_gradient.confidence

    @patch('src.backend.services.homing_algorithm._debug_mode_enabled', True)
    def test_doppler_compensation_debug_logging(self, homing_algorithm, platform_velocity, caplog):
        """Test that Doppler compensation debug information is logged correctly."""
        # Arrange
        homing_algorithm.set_platform_velocity(platform_velocity)
        homing_algorithm.set_signal_frequency(406_000_000)
        
        original_gradient = GradientVector(
            magnitude=0.5,
            direction=0.0,  # Due North
            confidence=0.8
        )
        
        # Act
        with caplog.at_level('DEBUG'):
            compensated_gradient = homing_algorithm._apply_doppler_compensation(original_gradient)
        
        # Assert - Debug logging occurred
        assert "Doppler compensation applied" in caplog.text
        assert "freq_shift=" in caplog.text
        assert "ratio=" in caplog.text
        assert compensated_gradient is not None

    def test_doppler_compensation_error_handling(self, homing_algorithm, platform_velocity):
        """Test error handling in Doppler compensation."""
        # Arrange - Set up invalid conditions that might cause errors
        homing_algorithm.set_platform_velocity(platform_velocity)
        # Set an invalid frequency to potentially cause calculation errors
        homing_algorithm._signal_frequency_hz = 0  # Invalid frequency
        
        original_gradient = GradientVector(
            magnitude=0.5,
            direction=45.0,
            confidence=0.8
        )
        
        # Act - Should handle errors gracefully
        compensated_gradient = homing_algorithm._apply_doppler_compensation(original_gradient)
        
        # Assert - Fallback to original gradient on error
        assert compensated_gradient is not None
        # Should return original or safely modified gradient, not crash