"""Unit tests for HomingAlgorithm service.

Tests RSSI gradient climbing and homing behavior per PRD-FR4.
"""

import pytest

from src.backend.services.homing_algorithm import HomingAlgorithm


class TestHomingAlgorithm:
    """Test RSSI gradient climbing homing algorithm."""

    @pytest.fixture
    def homing_algorithm(self):
        """Provide HomingAlgorithm instance."""
        return HomingAlgorithm()

    def test_homing_algorithm_initialization(self, homing_algorithm):
        """Test HomingAlgorithm initializes with correct defaults."""
        assert homing_algorithm.gradient_threshold > 0
        assert homing_algorithm.velocity_scale_factor > 0
        assert isinstance(homing_algorithm.rssi_history, list)

    def test_compute_rssi_gradient(self, homing_algorithm):
        """Test RSSI gradient computation from history."""
        # Add test RSSI readings
        rssi_values = [-60, -55, -50, -45, -40]  # Increasing signal strength
        for rssi in rssi_values:
            homing_algorithm.add_rssi_reading(rssi)

        gradient = homing_algorithm.compute_gradient()

        # Should show positive gradient (signal getting stronger)
        assert gradient > 0

    def test_compute_velocity_commands(self, homing_algorithm):
        """Test velocity command generation based on gradient."""
        # Set up strong positive gradient
        homing_algorithm.add_rssi_reading(-50)
        homing_algorithm.add_rssi_reading(-45)
        homing_algorithm.add_rssi_reading(-40)

        forward_velocity, yaw_rate = homing_algorithm.compute_velocity_commands()

        assert forward_velocity > 0  # Should move forward
        assert isinstance(yaw_rate, float)

    def test_approach_velocity_scaling(self, homing_algorithm):
        """Test velocity scaling based on signal strength."""
        # Weak signal should use higher velocity
        weak_velocity = homing_algorithm.scale_velocity_by_rssi(-70)

        # Strong signal should use lower velocity
        strong_velocity = homing_algorithm.scale_velocity_by_rssi(-30)

        assert weak_velocity > strong_velocity

    def test_circular_holding_pattern(self, homing_algorithm):
        """Test circular holding pattern when signal plateaus."""
        # Add plateaued signal readings
        for _ in range(5):
            homing_algorithm.add_rssi_reading(-25)  # Very strong, stable signal

        is_holding = homing_algorithm.should_enter_holding_pattern()

        assert is_holding is True

    def test_sampling_maneuvers(self, homing_algorithm):
        """Test S-turn sampling when gradient is unclear."""
        # Add confusing signal readings
        homing_algorithm.add_rssi_reading(-50)
        homing_algorithm.add_rssi_reading(-55)
        homing_algorithm.add_rssi_reading(-52)

        should_sample = homing_algorithm.should_perform_sampling()

        assert isinstance(should_sample, bool)

    def test_rssi_history_management(self, homing_algorithm):
        """Test RSSI history buffer management."""
        # Add more readings than buffer size
        for i in range(50):
            homing_algorithm.add_rssi_reading(-60 + i)

        # Should maintain reasonable history size
        assert len(homing_algorithm.rssi_history) <= 20

    def test_gradient_smoothing(self, homing_algorithm):
        """Test gradient smoothing to reduce noise."""
        # Add noisy RSSI readings
        noisy_values = [-50, -45, -55, -40, -60, -35]
        for rssi in noisy_values:
            homing_algorithm.add_rssi_reading(rssi)

        smooth_gradient = homing_algorithm.compute_smoothed_gradient()
        raw_gradient = homing_algorithm.compute_gradient()

        # Smoothed should be different from raw
        assert abs(smooth_gradient - raw_gradient) > 0
