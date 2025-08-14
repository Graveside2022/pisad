"""Unit tests for RSSI gradient-based homing algorithm."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.services.homing_algorithm import (
    GradientVector,
    HomingAlgorithm,
    HomingSubstage,
)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    mock = MagicMock()
    mock.homing.HOMING_FORWARD_VELOCITY_MAX = 5.0
    mock.homing.HOMING_YAW_RATE_MAX = 0.5
    mock.homing.HOMING_APPROACH_VELOCITY = 1.0
    mock.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
    mock.homing.HOMING_GRADIENT_MIN_SNR = 10.0
    mock.homing.HOMING_SAMPLING_TURN_RADIUS = 10.0
    mock.homing.HOMING_SAMPLING_DURATION = 5.0
    mock.homing.HOMING_APPROACH_THRESHOLD = -50.0
    mock.homing.HOMING_PLATEAU_VARIANCE = 2.0
    mock.homing.HOMING_VELOCITY_SCALE_FACTOR = 0.1
    return mock


@pytest.fixture
def homing_algorithm(mock_config):
    """Create homing algorithm instance with mocked config."""
    with patch("backend.services.homing_algorithm.get_config") as mock_get_config:
        mock_get_config.return_value = mock_config
        return HomingAlgorithm()


class TestHomingAlgorithm:
    """Test suite for HomingAlgorithm class."""

    def test_initialization(self, homing_algorithm):
        """Test algorithm initialization with config."""
        assert homing_algorithm.forward_velocity_max == 5.0
        assert homing_algorithm.yaw_rate_max == 0.5
        assert homing_algorithm.gradient_window_size == 10
        assert homing_algorithm.current_substage == HomingSubstage.IDLE
        assert len(homing_algorithm.rssi_history) == 0

    def test_add_rssi_sample(self, homing_algorithm):
        """Test adding RSSI samples to history buffer."""
        homing_algorithm.add_rssi_sample(-70.0, 10.0, 20.0, 45.0, 1.0)
        assert len(homing_algorithm.rssi_history) == 1

        sample = homing_algorithm.rssi_history[0]
        assert sample.rssi == -70.0
        assert sample.position_x == 10.0
        assert sample.position_y == 20.0
        assert sample.heading == 45.0
        assert sample.timestamp == 1.0

    def test_history_buffer_max_length(self, homing_algorithm):
        """Test that history buffer respects max length."""
        for i in range(15):
            homing_algorithm.add_rssi_sample(-70.0 + i, float(i), float(i), 0.0, float(i))

        assert len(homing_algorithm.rssi_history) == 10  # Max window size
        assert homing_algorithm.rssi_history[0].timestamp == 5.0  # Oldest kept

    def test_calculate_gradient_insufficient_samples(self, homing_algorithm):
        """Test gradient calculation with insufficient samples."""
        homing_algorithm.add_rssi_sample(-70.0, 0.0, 0.0, 0.0, 1.0)
        homing_algorithm.add_rssi_sample(-68.0, 1.0, 0.0, 0.0, 2.0)

        gradient = homing_algorithm.calculate_gradient()
        assert gradient is None

    def test_calculate_gradient_linear_increase(self, homing_algorithm):
        """Test gradient calculation with linear signal increase."""
        # Create samples with linear RSSI increase toward positive X
        positions = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
        rssi_values = [-80, -75, -70, -65, -60]  # 0.5 dB/m gradient

        for i, ((x, y), rssi) in enumerate(zip(positions, rssi_values, strict=False)):
            homing_algorithm.add_rssi_sample(rssi, float(x), float(y), 0.0, float(i))

        gradient = homing_algorithm.calculate_gradient()
        assert gradient is not None
        assert gradient.magnitude > 0.4  # Should be close to 0.5 dB/m
        assert gradient.magnitude < 0.6
        assert abs(gradient.direction - 0) < 10  # Should point toward positive X (0°)
        assert gradient.confidence >= 50  # Should have reasonable confidence

    def test_calculate_gradient_diagonal(self, homing_algorithm):
        """Test gradient calculation with diagonal signal increase."""
        # Create samples with signal increasing toward northeast
        positions = [(0, 0), (10, 10), (20, 20), (0, 10), (10, 0)]
        rssi_values = [-80, -70, -60, -75, -75]

        for i, ((x, y), rssi) in enumerate(zip(positions, rssi_values, strict=False)):
            homing_algorithm.add_rssi_sample(rssi, float(x), float(y), 0.0, float(i))

        gradient = homing_algorithm.calculate_gradient()
        assert gradient is not None
        assert gradient.magnitude > 0
        # Should point roughly northeast (45°)
        assert 30 < gradient.direction < 60

    def test_calculate_gradient_insufficient_spatial_diversity(self, homing_algorithm):
        """Test gradient with samples too close together."""
        # All samples at nearly same position
        for i in range(5):
            homing_algorithm.add_rssi_sample(-70.0 + i, 0.01 * i, 0.01 * i, 0.0, float(i))

        gradient = homing_algorithm.calculate_gradient()
        assert gradient is None  # Should fail due to insufficient spatial diversity

    def test_compute_optimal_heading(self, homing_algorithm):
        """Test optimal heading computation."""
        gradient = GradientVector(magnitude=1.0, direction=135.0, confidence=80.0)
        optimal = homing_algorithm.compute_optimal_heading(gradient)
        assert optimal == 135.0  # Should follow gradient direction

    def test_scale_velocity_by_gradient(self, homing_algorithm):
        """Test velocity scaling based on gradient."""
        # Strong gradient with high confidence
        gradient = GradientVector(magnitude=10.0, direction=0.0, confidence=90.0)
        velocity = homing_algorithm.scale_velocity_by_gradient(gradient)
        assert velocity > 0
        assert velocity <= homing_algorithm.forward_velocity_max

        # Weak gradient with low confidence
        gradient = GradientVector(magnitude=0.5, direction=0.0, confidence=30.0)
        velocity = homing_algorithm.scale_velocity_by_gradient(gradient)
        assert velocity < 1.0  # Should be slow with weak/uncertain gradient

    def test_calculate_yaw_rate(self, homing_algorithm):
        """Test yaw rate calculation."""
        # 90 degree turn to the right
        yaw_rate = homing_algorithm.calculate_yaw_rate(0.0, 90.0)
        assert yaw_rate > 0  # Positive for right turn
        assert yaw_rate <= homing_algorithm.yaw_rate_max

        # 90 degree turn to the left
        yaw_rate = homing_algorithm.calculate_yaw_rate(0.0, 270.0)
        assert yaw_rate < 0  # Negative for left turn
        assert yaw_rate >= -homing_algorithm.yaw_rate_max

        # No turn needed
        yaw_rate = homing_algorithm.calculate_yaw_rate(45.0, 45.0)
        assert abs(yaw_rate) < 0.01

    def test_generate_velocity_command_gradient_climb(self, homing_algorithm):
        """Test velocity command generation in gradient climb mode."""
        # Add samples for good gradient
        positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        rssi_values = [-80, -75, -70, -65]

        for i, ((x, y), rssi) in enumerate(zip(positions, rssi_values, strict=False)):
            homing_algorithm.add_rssi_sample(rssi, float(x), float(y), 0.0, float(i))

        gradient = homing_algorithm.calculate_gradient()
        command = homing_algorithm.generate_velocity_command(gradient, 45.0, 5.0)

        assert homing_algorithm.current_substage == HomingSubstage.GRADIENT_CLIMB
        assert command.forward_velocity > 0
        assert abs(command.yaw_rate) <= homing_algorithm.yaw_rate_max

    def test_generate_velocity_command_approach_mode(self, homing_algorithm):
        """Test velocity command in approach mode (high RSSI)."""
        # Add sample with RSSI above approach threshold
        homing_algorithm.add_rssi_sample(-45.0, 0.0, 0.0, 0.0, 1.0)
        homing_algorithm.add_rssi_sample(-44.0, 1.0, 0.0, 0.0, 2.0)
        homing_algorithm.add_rssi_sample(-43.0, 2.0, 0.0, 0.0, 3.0)

        gradient = homing_algorithm.calculate_gradient()
        command = homing_algorithm.generate_velocity_command(gradient, 0.0, 4.0)

        assert homing_algorithm.current_substage == HomingSubstage.APPROACH
        assert command.forward_velocity == homing_algorithm.approach_velocity

    def test_generate_velocity_command_sampling_mode(self, homing_algorithm):
        """Test velocity command in sampling mode (poor gradient)."""
        # No gradient available
        command = homing_algorithm.generate_velocity_command(None, 0.0, 1.0)

        assert homing_algorithm.current_substage == HomingSubstage.SAMPLING
        assert command.forward_velocity > 0
        assert homing_algorithm.sampling_start_time == 1.0

        # Continue sampling
        command = homing_algorithm.generate_velocity_command(None, 0.0, 2.0)
        assert homing_algorithm.current_substage == HomingSubstage.SAMPLING
        assert abs(command.yaw_rate) > 0  # Should be turning

    def test_detect_plateau(self, homing_algorithm):
        """Test plateau detection for holding pattern."""
        # Add samples with low variance and high signal
        for i in range(10):
            rssi = -55.0 + np.random.normal(0, 0.5)  # Small variance
            homing_algorithm.add_rssi_sample(rssi, float(i), float(i), 0.0, float(i))

        assert homing_algorithm._detect_plateau() is True

        # Add samples with high variance
        homing_algorithm.rssi_history.clear()
        for i in range(10):
            rssi = -55.0 + np.random.normal(0, 5.0)  # Large variance
            homing_algorithm.add_rssi_sample(rssi, float(i), float(i), 0.0, float(i))

        assert homing_algorithm._detect_plateau() is False

    def test_generate_velocity_command_holding_pattern(self, homing_algorithm):
        """Test holding pattern command generation."""
        # Setup plateau condition
        for i in range(10):
            homing_algorithm.add_rssi_sample(-55.0, float(i), float(i), 0.0, float(i))

        gradient = homing_algorithm.calculate_gradient()
        command = homing_algorithm.generate_velocity_command(gradient, 0.0, 11.0)

        assert homing_algorithm.current_substage == HomingSubstage.HOLDING
        assert command.forward_velocity > 0
        assert command.yaw_rate > 0  # Should be circling

    def test_reset(self, homing_algorithm):
        """Test algorithm reset."""
        # Add some data
        homing_algorithm.add_rssi_sample(-70.0, 0.0, 0.0, 0.0, 1.0)
        homing_algorithm.current_substage = HomingSubstage.GRADIENT_CLIMB
        homing_algorithm.sampling_start_time = 5.0

        # Reset
        homing_algorithm.reset()

        assert len(homing_algorithm.rssi_history) == 0
        assert homing_algorithm.current_substage == HomingSubstage.IDLE
        assert homing_algorithm.sampling_start_time is None
        assert homing_algorithm.last_gradient is None

    def test_get_status(self, homing_algorithm):
        """Test status reporting."""
        status = homing_algorithm.get_status()
        assert status["substage"] == "IDLE"
        assert status["sample_count"] == 0
        assert status["gradient_confidence"] == 0

        # Add samples and calculate gradient
        for i in range(5):
            homing_algorithm.add_rssi_sample(-70.0 + i, float(i * 10), 0.0, 0.0, float(i))

        homing_algorithm.calculate_gradient()
        status = homing_algorithm.get_status()

        assert status["sample_count"] == 5
        assert status["gradient_confidence"] > 0
        assert status["latest_rssi"] == -66.0

    def test_sampling_duration_timeout(self, homing_algorithm):
        """Test sampling maneuver timeout."""
        homing_algorithm.sampling_start_time = 1.0
        homing_algorithm.current_substage = HomingSubstage.SAMPLING

        # Before timeout
        command = homing_algorithm._generate_sampling_command(0.0, 3.0)
        assert abs(command.yaw_rate) > 0

        # After timeout
        command = homing_algorithm._generate_sampling_command(0.0, 7.0)
        assert homing_algorithm.sampling_start_time is None
        assert command.yaw_rate == 0.0

    def test_gradient_calculation_with_noise(self, homing_algorithm):
        """Test gradient calculation with noisy data."""
        # Linear increase with noise
        np.random.seed(42)
        for i in range(10):
            x = float(i * 10)
            y = float(i * 5)
            rssi = -80 + (i * 2) + np.random.normal(0, 1.0)  # Linear with noise
            homing_algorithm.add_rssi_sample(rssi, x, y, 0.0, float(i))

        gradient = homing_algorithm.calculate_gradient()
        assert gradient is not None
        assert gradient.magnitude > 0
        assert gradient.confidence >= 30  # With noise, confidence will be lower

    def test_yaw_rate_shortest_path(self, homing_algorithm):
        """Test yaw rate calculation takes shortest path."""
        # From 350° to 10° should turn right (not left all the way around)
        yaw_rate = homing_algorithm.calculate_yaw_rate(350.0, 10.0)
        assert yaw_rate > 0  # Positive for right turn
        assert abs(yaw_rate) < 0.2  # Should be small for 20° turn

        # From 10° to 350° should turn left
        yaw_rate = homing_algorithm.calculate_yaw_rate(10.0, 350.0)
        assert yaw_rate < 0  # Negative for left turn
        assert abs(yaw_rate) < 0.2  # Should be small for 20° turn
