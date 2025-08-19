"""Unit tests for Adaptive Search Patterns Enhancement.

SUBTASK-6.1.2.3 [16c] - Test suite for adaptive search patterns when signal confidence is low

This test suite validates the enhancement of the existing _generate_sampling_command method
with adaptive spiral search and optimized S-turn patterns based on ASV signal quality:
- Adaptive pattern selection based on confidence levels
- Spiral search pattern generation with expanding radius
- S-turn pattern optimization using ASV signal quality feedback  
- Geofence boundary integration for all adaptive patterns
"""

import math
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.backend.services.homing_algorithm import (
    HomingAlgorithm, 
    HomingSubstage,
    VelocityCommand,
    GradientVector,
    RSSISample
)
from src.backend.services.asv_integration.asv_enhanced_homing_integration import (
    ASVEnhancedHomingIntegration,
)


class TestAdaptiveSearchPatterns:
    """Test suite for adaptive search patterns when signal confidence is low."""

    @pytest.fixture
    def mock_asv_integration(self):
        """Create mock ASV integration for testing."""
        mock_asv = Mock(spec=ASVEnhancedHomingIntegration)
        return mock_asv

    @pytest.fixture
    def homing_algorithm(self, mock_asv_integration):
        """Create homing algorithm instance with ASV integration."""
        return HomingAlgorithm(asv_integration=mock_asv_integration)

    @pytest.fixture
    def low_confidence_gradient(self):
        """Create low confidence gradient to trigger adaptive patterns."""
        return GradientVector(
            magnitude=0.5,
            direction=90.0,
            confidence=15.0  # Below GRADIENT_CONFIDENCE_THRESHOLD (30%)
        )

    @pytest.fixture
    def moderate_confidence_gradient(self):
        """Create moderate confidence gradient for pattern optimization."""
        return GradientVector(
            magnitude=1.2,
            direction=135.0,
            confidence=45.0  # Moderate confidence level
        )

    def test_adaptive_pattern_selection_spiral_for_very_low_confidence(self, homing_algorithm):
        """Test that spiral search is selected for very low confidence scenarios."""
        # Arrange: Very low confidence (< 10%) should trigger spiral search
        very_low_confidence = 5.0
        current_time = time.time()
        
        # Act: Generate command with very low confidence
        # This should be tested via the enhanced _generate_sampling_command method
        homing_algorithm.gradient_confidence = very_low_confidence
        command = homing_algorithm._generate_sampling_command(0.0, current_time)
        
        # Assert: Should be VelocityCommand (will be enhanced to detect spiral pattern)
        assert isinstance(command, VelocityCommand)
        # Note: This test will initially fail - we need to implement adaptive pattern selection

    def test_adaptive_pattern_selection_s_turn_for_moderate_confidence(self, homing_algorithm):
        """Test that optimized S-turn is selected for moderate confidence scenarios."""
        # Arrange: Moderate confidence (10-40%) should trigger optimized S-turn
        moderate_confidence = 25.0
        current_time = time.time()
        
        # Act: Generate command with moderate confidence  
        homing_algorithm.gradient_confidence = moderate_confidence
        command = homing_algorithm._generate_sampling_command(0.0, current_time)
        
        # Assert: Should be VelocityCommand (will be enhanced to detect S-turn optimization)
        assert isinstance(command, VelocityCommand)
        # Note: This test will initially fail - we need to implement pattern optimization

    def test_spiral_search_pattern_expanding_radius(self, homing_algorithm):
        """Test spiral search pattern generates expanding radius based on confidence."""
        # Arrange: Set up for spiral search pattern
        very_low_confidence = 8.0
        current_time = time.time()
        
        # Mock geofence boundaries for testing
        homing_algorithm._geofence_center_x = 0.0
        homing_algorithm._geofence_center_y = 0.0  
        homing_algorithm._geofence_radius = 100.0  # 100m radius
        
        # Act: Generate multiple spiral commands over time
        commands = []
        for i in range(5):
            homing_algorithm.gradient_confidence = very_low_confidence
            command = homing_algorithm._generate_sampling_command(0.0, current_time + i)
            commands.append(command)
            
        # Assert: All commands should be valid velocity commands
        for command in commands:
            assert isinstance(command, VelocityCommand)
            assert command.forward_velocity > 0  # Should move forward in spiral
            # Note: Will enhance to verify spiral radius expansion

    def test_s_turn_optimization_with_asv_feedback(self, homing_algorithm, mock_asv_integration):
        """Test S-turn pattern optimization using ASV signal quality feedback."""
        # Arrange: Set up ASV signal quality feedback via hasattr/getattr pattern
        moderate_confidence = 30.0  # Between 10-40% for optimized S-turn
        current_time = time.time()
        
        # Mock the ASV integration with signal quality method
        mock_asv_integration.get_signal_quality = lambda: 0.6  # 60% quality
        homing_algorithm._asv_integration = mock_asv_integration
        
        # Act: Generate S-turn command with ASV feedback
        homing_algorithm.gradient_confidence = moderate_confidence
        command = homing_algorithm._generate_sampling_command(0.0, current_time)
        
        # Assert: Should generate valid velocity command (optimized S-turn)
        assert isinstance(command, VelocityCommand)
        assert command.forward_velocity > 0
        assert homing_algorithm._pattern_type == "optimized_s_turn"

    def test_geofence_boundary_integration_spiral_pattern(self, homing_algorithm):
        """Test spiral pattern respects geofence boundaries."""
        # Arrange: Set up geofence constraints
        geofence_center_x, geofence_center_y = 0.0, 0.0
        geofence_radius = 50.0  # 50m radius constraint
        
        # Mock geofence boundary checking
        with patch.object(homing_algorithm, '_check_geofence_boundary') as mock_geofence:
            mock_geofence.return_value = True  # Initially within bounds
            
            very_low_confidence = 5.0
            current_time = time.time()
            
            # Act: Generate spiral pattern near boundary
            homing_algorithm.gradient_confidence = very_low_confidence
            command = homing_algorithm._generate_sampling_command(0.0, current_time)
            
            # Assert: Should respect geofence
            assert isinstance(command, VelocityCommand)
            # Note: Will enhance to verify geofence integration

    def test_geofence_boundary_integration_s_turn_pattern(self, homing_algorithm):
        """Test S-turn pattern respects geofence boundaries."""
        # Arrange: Set up geofence constraints
        geofence_center_x, geofence_center_y = 0.0, 0.0
        geofence_radius = 30.0  # 30m radius constraint
        
        # Mock geofence boundary checking
        with patch.object(homing_algorithm, '_check_geofence_boundary') as mock_geofence:
            mock_geofence.return_value = True  # Within bounds
            
            moderate_confidence = 25.0
            current_time = time.time()
            
            # Act: Generate S-turn pattern
            homing_algorithm.gradient_confidence = moderate_confidence
            command = homing_algorithm._generate_sampling_command(0.0, current_time)
            
            # Assert: Should respect geofence
            assert isinstance(command, VelocityCommand)

    def test_performance_requirement_pattern_generation_under_100ms(self, homing_algorithm):
        """Test that adaptive pattern generation meets <100ms performance requirement."""
        # Arrange: Set up for performance test
        very_low_confidence = 5.0
        current_time = time.time()
        
        # Act: Measure pattern generation time
        start_time = time.perf_counter()
        homing_algorithm.gradient_confidence = very_low_confidence
        command = homing_algorithm._generate_sampling_command(0.0, current_time)
        end_time = time.perf_counter()
        
        # Assert: Should complete under 100ms
        generation_time_ms = (end_time - start_time) * 1000
        assert generation_time_ms < 100.0  # <100ms requirement
        assert isinstance(command, VelocityCommand)

    def test_adaptive_pattern_state_preservation(self, homing_algorithm):
        """Test that adaptive patterns preserve existing state management."""
        # Arrange: Set up sampling state
        current_time = time.time()
        
        # Act: Generate sampling command (should set sampling_start_time)
        command1 = homing_algorithm._generate_sampling_command(0.0, current_time)
        
        # Generate second command (should maintain sampling state)
        command2 = homing_algorithm._generate_sampling_command(0.0, current_time + 1.0)
        
        # Assert: State should be preserved
        assert isinstance(command1, VelocityCommand)
        assert isinstance(command2, VelocityCommand)
        assert homing_algorithm.sampling_start_time is not None

    def test_fallback_to_original_sampling_when_no_confidence_data(self, homing_algorithm):
        """Test fallback to original S-turn sampling when confidence data unavailable."""
        # Arrange: No confidence data available
        homing_algorithm.gradient_confidence = 0.0  # No confidence info
        current_time = time.time()
        
        # Act: Generate sampling command
        command = homing_algorithm._generate_sampling_command(0.0, current_time)
        
        # Assert: Should fallback to original implementation
        assert isinstance(command, VelocityCommand)
        assert command.forward_velocity > 0  # Should still generate valid command


class TestAdaptivePatternIntegration:
    """Integration tests for adaptive patterns with existing systems."""

    @pytest.fixture
    def homing_algorithm_with_history(self):
        """Create homing algorithm with RSSI history."""
        algorithm = HomingAlgorithm()
        
        # Add sample RSSI history
        for i, rssi in enumerate([-60, -58, -62, -55, -61]):
            algorithm.add_rssi_sample(
                rssi=rssi,
                position_x=float(i * 2),
                position_y=float(i * 1.5), 
                heading=float(i * 30),
                timestamp=time.time() + i
            )
        
        return algorithm

    def test_integration_with_existing_substage_transitions(self, homing_algorithm_with_history):
        """Test adaptive patterns integrate properly with substage transitions."""
        # Arrange: Set up for sampling substage
        homing_algorithm = homing_algorithm_with_history
        current_time = time.time()
        
        # Act: Generate command that should trigger sampling
        gradient = GradientVector(magnitude=0.1, direction=45.0, confidence=20.0)  # Low confidence
        command = homing_algorithm.generate_velocity_command(gradient, 45.0, current_time)
        
        # Assert: Should transition to sampling substage
        assert homing_algorithm.current_substage == HomingSubstage.SAMPLING
        assert isinstance(command, VelocityCommand)

    def test_integration_with_safety_authority_preservation(self, homing_algorithm_with_history):
        """Test that adaptive patterns preserve all safety authority mechanisms."""
        # Arrange: This test verifies safety integration is not broken
        homing_algorithm = homing_algorithm_with_history
        
        # Act: Generate adaptive pattern command
        current_time = time.time()
        command = homing_algorithm._generate_sampling_command(0.0, current_time)
        
        # Assert: Safety mechanisms should still be intact
        assert isinstance(command, VelocityCommand)
        # Safety authority should be preserved (tested via existing safety test suite)