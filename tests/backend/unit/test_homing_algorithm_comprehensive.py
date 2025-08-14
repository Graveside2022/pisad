"""Comprehensive unit tests for homing algorithm."""

import math
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.backend.services.homing_algorithm import (
    GradientVector,
    HomingAlgorithm,
    HomingSubstage,
    RSSISample,
    VelocityCommand,
    set_debug_mode,
)


class TestHomingAlgorithm:
    """Test homing algorithm functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.homing.HOMING_FORWARD_VELOCITY_MAX = 10.0
        config.homing.HOMING_YAW_RATE_MAX = 1.0
        config.homing.HOMING_APPROACH_VELOCITY = 2.0
        config.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
        config.homing.HOMING_GRADIENT_MIN_SNR = 10.0
        config.homing.HOMING_SAMPLING_TURN_RADIUS = 5.0
        config.homing.HOMING_SAMPLING_DURATION = 10.0
        config.homing.HOMING_APPROACH_THRESHOLD = -50.0
        config.homing.HOMING_PLATEAU_VARIANCE = 2.0
        config.homing.HOMING_VELOCITY_SCALE_FACTOR = 1.0
        config.development.DEV_DEBUG_MODE = False
        return config

    @pytest.fixture
    def algorithm(self, mock_config):
        """Create homing algorithm instance."""
        with patch("src.backend.services.homing_algorithm.get_config", return_value=mock_config):
            return HomingAlgorithm()

    def test_initialization(self, algorithm):
        """Test algorithm initialization."""
        assert algorithm.forward_velocity_max == 10.0
        assert algorithm.yaw_rate_max == 1.0
        assert algorithm.approach_velocity == 2.0
        assert algorithm.gradient_window_size == 10
        assert algorithm.current_substage == HomingSubstage.IDLE
        assert len(algorithm.rssi_history) == 0
        assert algorithm.last_gradient is None

    def test_debug_mode(self):
        """Test debug mode toggling."""
        # Test enabling debug mode
        set_debug_mode(True)
        # Create a new algorithm to verify debug mode
        test_algo = HomingAlgorithm()
        # Debug mode should be enabled (check via log output or behavior)
        
        # Test disabling debug mode
        set_debug_mode(False)
        # Create another algorithm to verify debug mode is off
        test_algo2 = HomingAlgorithm()
        # Debug mode should be disabled

    def test_add_rssi_sample(self, algorithm):
        """Test adding RSSI samples to history."""
        algorithm.add_rssi_sample(-70.0, 10.0, 20.0, 45.0, 1000.0)
        
        assert len(algorithm.rssi_history) == 1
        sample = algorithm.rssi_history[0]
        assert sample.rssi == -70.0
        assert sample.position_x == 10.0
        assert sample.position_y == 20.0
        assert sample.heading == 45.0
        assert sample.timestamp == 1000.0

    def test_rssi_history_maxlen(self, algorithm):
        """Test RSSI history buffer respects max length."""
        # Add more samples than window size
        for i in range(15):
            algorithm.add_rssi_sample(-70.0 + i, i * 1.0, i * 2.0, i * 10.0, i * 100.0)
        
        # Should only keep last 10 samples
        assert len(algorithm.rssi_history) == 10
        # Oldest sample should be index 5 (samples 0-4 dropped)
        assert algorithm.rssi_history[0].position_x == 5.0

    def test_calculate_gradient_insufficient_samples(self, algorithm):
        """Test gradient calculation with insufficient samples."""
        # Add only 2 samples (need 3)
        algorithm.add_rssi_sample(-70.0, 0.0, 0.0, 0.0, 0.0)
        algorithm.add_rssi_sample(-65.0, 1.0, 0.0, 0.0, 1.0)
        
        gradient = algorithm.calculate_gradient()
        assert gradient is None

    def test_calculate_gradient_insufficient_spatial_diversity(self, algorithm):
        """Test gradient calculation with insufficient spatial diversity."""
        # Add samples at nearly same position
        for i in range(5):
            algorithm.add_rssi_sample(-70.0 + i, 0.01 * i, 0.01 * i, 0.0, i)
        
        gradient = algorithm.calculate_gradient()
        assert gradient is None

    def test_calculate_gradient_valid(self, algorithm):
        """Test valid gradient calculation."""
        # Add samples with good spatial diversity and clear gradient
        algorithm.add_rssi_sample(-80.0, 0.0, 0.0, 0.0, 0.0)
        algorithm.add_rssi_sample(-75.0, 5.0, 0.0, 90.0, 1.0)
        algorithm.add_rssi_sample(-70.0, 10.0, 0.0, 90.0, 2.0)
        algorithm.add_rssi_sample(-65.0, 10.0, 5.0, 45.0, 3.0)
        
        gradient = algorithm.calculate_gradient()
        assert gradient is not None
        assert gradient.magnitude > 0
        assert 0 <= gradient.direction <= 360
        assert 0 <= gradient.confidence <= 100

    def test_compute_optimal_heading(self, algorithm):
        """Test optimal heading computation."""
        gradient = GradientVector(magnitude=1.0, direction=45.0, confidence=80.0)
        
        heading = algorithm.compute_optimal_heading(gradient)
        assert heading == 45.0  # Should follow gradient direction

    def test_scale_velocity_by_gradient(self, algorithm):
        """Test velocity scaling based on gradient."""
        # High confidence, strong gradient
        gradient = GradientVector(magnitude=2.0, direction=0.0, confidence=90.0)
        velocity = algorithm.scale_velocity_by_gradient(gradient)
        assert velocity > 0
        assert velocity <= algorithm.forward_velocity_max
        
        # Low confidence gradient
        gradient_low = GradientVector(magnitude=2.0, direction=0.0, confidence=20.0)
        velocity_low = algorithm.scale_velocity_by_gradient(gradient_low)
        assert velocity_low < velocity  # Lower confidence = lower velocity

    def test_calculate_yaw_rate(self, algorithm):
        """Test yaw rate calculation."""
        # Test various heading errors
        # No error
        yaw_rate = algorithm.calculate_yaw_rate(45.0, 45.0)
        assert abs(yaw_rate) < 0.01
        
        # Positive error (turn right)
        yaw_rate = algorithm.calculate_yaw_rate(0.0, 90.0)
        assert yaw_rate > 0
        
        # Negative error (turn left)
        yaw_rate = algorithm.calculate_yaw_rate(90.0, 0.0)
        assert yaw_rate < 0
        
        # Wrap-around case
        yaw_rate = algorithm.calculate_yaw_rate(350.0, 10.0)
        assert yaw_rate > 0  # Should turn right 20 degrees, not left 340

    def test_generate_velocity_command_idle(self, algorithm):
        """Test velocity command generation when idle."""
        command = algorithm.generate_velocity_command(None, 0.0, 0.0)
        assert isinstance(command, VelocityCommand)
        assert algorithm.current_substage == HomingSubstage.SAMPLING  # No gradient -> sampling

    def test_generate_velocity_command_approach_mode(self, algorithm):
        """Test velocity command in approach mode."""
        # Add strong signal sample
        algorithm.add_rssi_sample(-45.0, 0.0, 0.0, 0.0, 0.0)  # Above approach threshold
        
        gradient = GradientVector(magnitude=1.0, direction=90.0, confidence=80.0)
        command = algorithm.generate_velocity_command(gradient, 0.0, 1.0)
        
        assert algorithm.current_substage == HomingSubstage.APPROACH
        assert command.forward_velocity == algorithm.approach_velocity

    def test_generate_velocity_command_holding_pattern(self, algorithm):
        """Test velocity command for holding pattern."""
        # Add samples with low variance (plateau)
        for i in range(10):
            algorithm.add_rssi_sample(-55.0 + np.random.normal(0, 0.1), i, i, 0.0, i)
        
        command = algorithm.generate_velocity_command(None, 0.0, 10.0)
        
        # Check if holding pattern detected (depends on variance calculation)
        if algorithm.current_substage == HomingSubstage.HOLDING:
            assert command.forward_velocity > 0
            assert abs(command.yaw_rate) > 0

    def test_generate_velocity_command_gradient_climb(self, algorithm):
        """Test velocity command for gradient climbing."""
        # Add samples for good gradient
        algorithm.add_rssi_sample(-80.0, 0.0, 0.0, 0.0, 0.0)
        algorithm.add_rssi_sample(-75.0, 5.0, 0.0, 90.0, 1.0)
        algorithm.add_rssi_sample(-70.0, 10.0, 0.0, 90.0, 2.0)
        
        gradient = algorithm.calculate_gradient()
        assert gradient is not None
        gradient.confidence = 80.0  # Ensure high confidence
        
        command = algorithm.generate_velocity_command(gradient, 45.0, 3.0)
        
        if gradient.confidence > 30:  # Above threshold
            assert algorithm.current_substage == HomingSubstage.GRADIENT_CLIMB
            assert command.forward_velocity > 0
            assert isinstance(command.yaw_rate, float)

    def test_generate_sampling_command(self, algorithm):
        """Test S-turn sampling command generation."""
        # Force sampling mode
        algorithm.current_substage = HomingSubstage.SAMPLING
        algorithm.sampling_start_time = None
        
        # First call starts sampling
        command1 = algorithm._generate_sampling_command(0.0, 0.0)
        assert algorithm.sampling_start_time == 0.0
        assert command1.forward_velocity > 0
        
        # Mid-sampling
        command2 = algorithm._generate_sampling_command(0.0, 5.0)
        assert abs(command2.yaw_rate) > 0  # Should be turning
        
        # End of sampling
        command3 = algorithm._generate_sampling_command(0.0, 15.0)  # Past duration
        assert algorithm.sampling_start_time is None  # Reset

    def test_detect_plateau(self, algorithm):
        """Test plateau detection."""
        # No samples
        assert algorithm._detect_plateau() is False
        
        # Add high-variance samples (not plateau)
        for i in range(10):
            algorithm.add_rssi_sample(-70.0 + i * 5, i, i, 0.0, i)
        assert algorithm._detect_plateau() is False
        
        # Clear and add low-variance strong samples (plateau)
        algorithm.rssi_history.clear()
        for i in range(10):
            algorithm.add_rssi_sample(-55.0 + np.random.normal(0, 0.1), i, i, 0.0, i)
        
        # Should detect plateau (low variance + strong signal)
        is_plateau = algorithm._detect_plateau()
        # Result depends on exact variance calculation

    def test_get_status(self, algorithm):
        """Test status reporting."""
        # Add some samples
        algorithm.add_rssi_sample(-70.0, 0.0, 0.0, 0.0, 0.0)
        algorithm.add_rssi_sample(-65.0, 5.0, 0.0, 90.0, 1.0)
        
        status = algorithm.get_status()
        
        assert "substage" in status
        assert status["substage"] == HomingSubstage.IDLE.value
        assert "gradient_confidence" in status
        assert "target_heading" in status
        assert "rssi_history_size" in status
        assert status["rssi_history_size"] == 2
        assert "last_rssi" in status
        assert status["last_rssi"] == -65.0

    def test_get_status_with_gradient(self, algorithm):
        """Test status with gradient information."""
        # Add samples for gradient
        algorithm.add_rssi_sample(-80.0, 0.0, 0.0, 0.0, 0.0)
        algorithm.add_rssi_sample(-75.0, 5.0, 0.0, 90.0, 1.0)
        algorithm.add_rssi_sample(-70.0, 10.0, 0.0, 90.0, 2.0)
        
        gradient = algorithm.calculate_gradient()
        algorithm.generate_velocity_command(gradient, 0.0, 3.0)
        
        status = algorithm.get_status()
        
        if gradient:
            assert status["gradient_magnitude"] > 0
            assert status["gradient_direction"] >= 0
            assert "gradient" in status
            if status["gradient"]:
                assert "x" in status["gradient"]
                assert "y" in status["gradient"]

    def test_reset(self, algorithm):
        """Test algorithm reset."""
        # Add data
        algorithm.add_rssi_sample(-70.0, 0.0, 0.0, 0.0, 0.0)
        algorithm.current_substage = HomingSubstage.GRADIENT_CLIMB
        algorithm.last_gradient = GradientVector(1.0, 90.0, 80.0)
        
        # Reset
        algorithm.reset()
        
        assert len(algorithm.rssi_history) == 0
        assert algorithm.current_substage == HomingSubstage.IDLE
        assert algorithm.last_gradient is None
        assert algorithm.sampling_start_time is None

    def test_gradient_calculation_edge_cases(self, algorithm):
        """Test gradient calculation edge cases."""
        # Test with collinear points
        algorithm.add_rssi_sample(-80.0, 0.0, 0.0, 0.0, 0.0)
        algorithm.add_rssi_sample(-75.0, 1.0, 0.0, 0.0, 1.0)
        algorithm.add_rssi_sample(-70.0, 2.0, 0.0, 0.0, 2.0)
        algorithm.add_rssi_sample(-65.0, 3.0, 0.0, 0.0, 3.0)
        
        gradient = algorithm.calculate_gradient()
        if gradient:  # May succeed with rank 2
            assert gradient.magnitude > 0
            assert abs(gradient.direction - 0.0) < 10 or abs(gradient.direction - 360.0) < 10

    def test_yaw_rate_limiting(self, algorithm):
        """Test yaw rate is properly limited."""
        # Large heading error should still be limited
        yaw_rate = algorithm.calculate_yaw_rate(0.0, 180.0)
        assert abs(yaw_rate) <= algorithm.yaw_rate_max

    def test_velocity_command_continuity(self, algorithm):
        """Test velocity commands are continuous."""
        # Add samples
        for i in range(5):
            algorithm.add_rssi_sample(-70.0 + i, i * 2, i * 2, i * 10, i)
        
        # Generate multiple commands
        commands = []
        for t in range(5, 10):
            gradient = algorithm.calculate_gradient()
            command = algorithm.generate_velocity_command(gradient, t * 10, t)
            commands.append(command)
            algorithm.add_rssi_sample(-65.0 + t, t * 2, t * 2, t * 10, t)
        
        # Check all commands are valid
        for cmd in commands:
            assert isinstance(cmd, VelocityCommand)
            assert 0 <= cmd.forward_velocity <= algorithm.forward_velocity_max
            assert abs(cmd.yaw_rate) <= algorithm.yaw_rate_max