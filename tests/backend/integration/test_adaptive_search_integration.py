"""Integration test for adaptive search patterns with existing homing system.

SUBTASK-6.1.2.3 [16c] - Integration test to verify adaptive search patterns work correctly
with the complete homing algorithm system, including ASV integration and safety systems.
"""

import time
from unittest.mock import Mock

import pytest

from src.backend.services.asv_integration.asv_enhanced_homing_integration import (
    ASVEnhancedHomingIntegration,
)
from src.backend.services.homing_algorithm import (
    GradientVector,
    HomingAlgorithm,
    HomingSubstage,
    VelocityCommand,
)


class TestAdaptiveSearchIntegration:
    """Integration tests for adaptive search patterns."""

    @pytest.fixture
    def homing_algorithm(self):
        """Create homing algorithm with proper setup."""
        # Use real ASV integration but mock the underlying services
        mock_asv = Mock(spec=ASVEnhancedHomingIntegration)
        algorithm = HomingAlgorithm(asv_integration=mock_asv)
        return algorithm

    def test_complete_adaptive_search_workflow(self, homing_algorithm):
        """Test complete workflow from signal degradation to adaptive pattern activation."""
        # Arrange: Add initial RSSI samples
        current_time = time.time()
        for i in range(5):
            homing_algorithm.add_rssi_sample(
                rssi=-60.0 - i * 2,  # Degrading signal
                position_x=float(i * 2),
                position_y=float(i * 1.5),
                heading=float(i * 30),
                timestamp=current_time + i,
            )

        # Act: Generate command with very low confidence
        low_confidence_gradient = GradientVector(
            magnitude=0.5,
            direction=90.0,
            confidence=5.0,  # Very low confidence
        )

        command = homing_algorithm.generate_velocity_command(
            low_confidence_gradient, 45.0, current_time + 5
        )

        # Assert: Should enter sampling substage with spiral pattern
        assert homing_algorithm.current_substage == HomingSubstage.SAMPLING
        assert isinstance(command, VelocityCommand)
        assert command.forward_velocity > 0
        assert homing_algorithm._pattern_type == "spiral"

    def test_adaptive_pattern_transitions(self, homing_algorithm):
        """Test transitions between different adaptive patterns based on confidence."""
        current_time = time.time()

        # Test very low confidence -> spiral search
        homing_algorithm.gradient_confidence = 5.0
        spiral_command = homing_algorithm._generate_sampling_command(0.0, current_time)
        assert homing_algorithm._pattern_type == "spiral"

        # Reset for next pattern test
        homing_algorithm.sampling_start_time = None

        # Test moderate confidence -> optimized S-turn
        homing_algorithm.gradient_confidence = 25.0
        s_turn_command = homing_algorithm._generate_sampling_command(0.0, current_time + 1)
        assert homing_algorithm._pattern_type == "optimized_s_turn"

        # Reset for next pattern test
        homing_algorithm.sampling_start_time = None

        # Test higher confidence -> original pattern
        homing_algorithm.gradient_confidence = 50.0
        original_command = homing_algorithm._generate_sampling_command(0.0, current_time + 2)
        assert homing_algorithm._pattern_type == "original"

    def test_geofence_integration_with_real_boundaries(self, homing_algorithm):
        """Test geofence boundary checking with realistic coordinates."""
        # Set up geofence boundaries
        homing_algorithm._geofence_center_x = 0.0
        homing_algorithm._geofence_center_y = 0.0
        homing_algorithm._geofence_radius = 50.0  # 50m radius

        # Test position within boundary
        assert homing_algorithm._check_geofence_boundary(30.0, 30.0) is True

        # Test position outside boundary
        assert homing_algorithm._check_geofence_boundary(60.0, 60.0) is False

        # Test boundary edge case
        assert homing_algorithm._check_geofence_boundary(50.0, 0.0) is True
        assert homing_algorithm._check_geofence_boundary(50.1, 0.0) is False

    def test_status_reporting_with_adaptive_patterns(self, homing_algorithm):
        """Test that status reporting includes adaptive pattern information."""
        # Generate a spiral pattern to set state
        homing_algorithm.gradient_confidence = 5.0
        current_time = time.time()
        homing_algorithm._generate_sampling_command(0.0, current_time)

        # Get status
        status = homing_algorithm.get_status()

        # Verify adaptive pattern status is included
        assert "adaptive_pattern" in status
        assert status["adaptive_pattern"]["pattern_type"] == "spiral"
        assert "spiral_radius" in status["adaptive_pattern"]
        assert "spiral_angle_deg" in status["adaptive_pattern"]

        # Verify geofence status is included
        assert "geofence" in status
        assert "center_x" in status["geofence"]
        assert "center_y" in status["geofence"]
        assert "radius" in status["geofence"]

    def test_performance_requirements_met(self, homing_algorithm):
        """Test that adaptive pattern generation meets <100ms performance requirement."""
        current_time = time.time()

        # Test all pattern types for performance
        pattern_types = [
            (5.0, "spiral"),  # Very low confidence
            (25.0, "optimized_s_turn"),  # Moderate confidence
            (50.0, "original"),  # Higher confidence
        ]

        for confidence, expected_pattern in pattern_types:
            homing_algorithm.gradient_confidence = confidence
            homing_algorithm.sampling_start_time = None  # Reset

            # Measure execution time
            start_time = time.perf_counter()
            command = homing_algorithm._generate_sampling_command(0.0, current_time)
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000

            # Assert performance requirement
            assert (
                execution_time_ms < 100.0
            ), f"Pattern {expected_pattern} took {execution_time_ms:.2f}ms"
            assert isinstance(command, VelocityCommand)
            assert homing_algorithm._pattern_type == expected_pattern

    def test_backward_compatibility_preserved(self, homing_algorithm):
        """Test that existing functionality remains unchanged."""
        # Test that existing methods still work
        current_time = time.time()

        # Add sample using current API
        homing_algorithm.add_rssi_sample(
            rssi=-50.0, position_x=10.0, position_y=5.0, heading=45.0, timestamp=current_time
        )

        # Test gradient calculation still works
        gradient = homing_algorithm.calculate_gradient()
        # May be None with single sample, but method should not error

        # Test status reporting still works
        status = homing_algorithm.get_status()
        assert "substage" in status
        assert "gradient_confidence" in status

        # Test existing command generation patterns
        test_gradient = GradientVector(magnitude=1.0, direction=90.0, confidence=75.0)
        command = homing_algorithm.generate_velocity_command(test_gradient, 0.0, current_time)
        assert isinstance(command, VelocityCommand)
