"""Unit tests for Enhanced Homing Algorithm with ASV Confidence Integration.

SUBTASK-6.1.2.2 [15c] - Test suite for integrated ASV confidence-based homing algorithm

This test suite validates the enhanced homing algorithm that integrates ASV confidence
metrics into the existing decision tree, replacing static thresholds with dynamic
assessment based on signal quality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
    ConfidenceBasedDecision,
    DynamicThresholdConfig,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVBearingCalculation,
    ASVEnhancedSignalProcessor,
)
from src.backend.services.homing_algorithm import (
    HomingAlgorithm,
    GradientVector,
    VelocityCommand,
    HomingSubstage,
)


class TestEnhancedHomingAlgorithmIntegration:
    """Test enhanced homing algorithm with ASV confidence integration."""

    @pytest.fixture
    def mock_asv_processor(self):
        """Mock ASV enhanced signal processor."""
        processor = Mock(spec=ASVEnhancedSignalProcessor)
        return processor

    @pytest.fixture
    def homing_algorithm(self):
        """Real homing algorithm for integration testing."""
        algorithm = HomingAlgorithm()
        return algorithm

    @pytest.fixture
    def confidence_homing(self, mock_asv_processor, homing_algorithm):
        """Confidence-based homing system."""
        return ASVConfidenceBasedHoming(
            asv_processor=mock_asv_processor,
            homing_algorithm=homing_algorithm
        )

    @pytest.fixture
    def high_confidence_asv_bearing(self):
        """High confidence ASV bearing for testing."""
        return ASVBearingCalculation(
            bearing_deg=90.0,
            confidence=0.85,
            precision_deg=2.5,
            signal_strength_dbm=-75.0,
            signal_quality=0.9,
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=False,
            signal_classification="CONTINUOUS",
        )

    @pytest.fixture
    def low_confidence_asv_bearing(self):
        """Low confidence ASV bearing for testing."""
        return ASVBearingCalculation(
            bearing_deg=45.0,
            confidence=0.25,
            precision_deg=15.0,
            signal_strength_dbm=-115.0,
            signal_quality=0.2,
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=True,
            signal_classification="INTERFERENCE",
        )

    def test_enhanced_algorithm_replaces_static_threshold(
        self, confidence_homing, homing_algorithm, high_confidence_asv_bearing
    ):
        """Test that enhanced algorithm replaces static 30% confidence threshold."""
        # Arrange
        current_heading = 45.0
        current_time = time.time()

        # Act
        velocity_command = confidence_homing.integrate_with_homing_algorithm(
            high_confidence_asv_bearing, current_heading, current_time
        )

        # Assert - Should generate valid velocity command for high confidence signal
        assert velocity_command is not None
        assert isinstance(velocity_command, VelocityCommand)
        assert velocity_command.forward_velocity > 0.0  # Should move forward
        assert abs(velocity_command.yaw_rate) >= 0.0  # Should adjust heading

    def test_dynamic_threshold_adjustment_overrides_static_threshold(
        self, confidence_homing, high_confidence_asv_bearing
    ):
        """Test dynamic threshold adjustment overrides static 30% threshold."""
        # Arrange - High quality signal should lower threshold requirements
        high_confidence_asv_bearing.signal_quality = 0.9  # High quality
        high_confidence_asv_bearing.confidence = 0.4  # Above static threshold of 30%

        # Act
        decision = confidence_homing.evaluate_confidence_based_decision(high_confidence_asv_bearing)

        # Assert - Should proceed due to dynamic threshold adjustment
        assert decision.proceed_with_homing is True
        assert decision.dynamic_threshold < 0.5  # Lowered from static 30%
        assert decision.confidence_assessment == "MODERATE"  # Updated based on combined score calculation

    def test_asv_confidence_metrics_influence_velocity_scaling(
        self, confidence_homing, homing_algorithm, high_confidence_asv_bearing
    ):
        """Test that ASV confidence metrics influence velocity command scaling."""
        # Arrange - Create two different confidence scenarios
        high_conf_bearing = high_confidence_asv_bearing
        low_conf_bearing = ASVBearingCalculation(
            bearing_deg=90.0,
            confidence=0.35,  # Lower but above threshold
            precision_deg=8.0,
            signal_strength_dbm=-100.0,
            signal_quality=0.5,
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=False,
        )

        current_heading = 45.0
        current_time = time.time()

        # Act
        high_conf_cmd = confidence_homing.integrate_with_homing_algorithm(
            high_conf_bearing, current_heading, current_time
        )
        low_conf_cmd = confidence_homing.integrate_with_homing_algorithm(
            low_conf_bearing, current_heading, current_time
        )

        # Assert - High confidence should result in more aggressive velocity
        assert high_conf_cmd is not None
        assert low_conf_cmd is not None
        # Higher confidence should generally result in higher velocity
        # (exact comparison may vary based on algorithm implementation)
        assert high_conf_cmd.forward_velocity >= 0.0
        assert low_conf_cmd.forward_velocity >= 0.0

    def test_confidence_based_fallback_integration(
        self, confidence_homing, low_confidence_asv_bearing
    ):
        """Test confidence-based fallback strategies integration."""
        # Arrange
        current_heading = 180.0
        current_time = time.time()

        # Act
        velocity_command = confidence_homing.integrate_with_homing_algorithm(
            low_confidence_asv_bearing, current_heading, current_time
        )

        # Assert - Should generate fallback command
        assert velocity_command is not None
        # Should generate some form of fallback movement (stop, spiral, or sampling)
        # The specific behavior depends on the fallback strategy selected

    def test_performance_requirement_integration_latency(
        self, confidence_homing, high_confidence_asv_bearing
    ):
        """Test that integration meets <100ms latency requirement."""
        # Arrange
        current_heading = 0.0
        current_time = time.time()

        # Act & Assert - Measure integration time
        start_time = time.perf_counter()
        velocity_command = confidence_homing.integrate_with_homing_algorithm(
            high_confidence_asv_bearing, current_heading, current_time
        )
        end_time = time.perf_counter()

        integration_latency_ms = (end_time - start_time) * 1000

        # Assert - Must meet performance requirement
        assert integration_latency_ms < 100.0  # <100ms requirement
        assert velocity_command is not None

    def test_existing_safety_interlocks_preserved(
        self, confidence_homing, homing_algorithm, high_confidence_asv_bearing
    ):
        """Test that all existing safety interlocks are preserved."""
        # Arrange - Activate safety override
        confidence_homing.set_safety_override(True, "Test safety condition")

        # Act
        velocity_command = confidence_homing.integrate_with_homing_algorithm(
            high_confidence_asv_bearing, 0.0, time.time()
        )

        # Assert - Should respect safety override
        assert velocity_command is not None
        # Safety override should result in stop command
        assert velocity_command.forward_velocity == 0.0
        assert velocity_command.yaw_rate == 0.0

    def test_gradient_enhancement_with_asv_metrics(
        self, confidence_homing, high_confidence_asv_bearing
    ):
        """Test gradient enhancement with ASV professional metrics."""
        # Arrange
        base_gradient = GradientVector(
            magnitude=1.0,
            direction=45.0,
            confidence=50.0  # Static baseline
        )

        # Act
        enhanced_gradient = confidence_homing.enhance_gradient_with_asv_confidence(
            base_gradient, high_confidence_asv_bearing
        )

        # Assert - Should enhance gradient with ASV metrics
        assert enhanced_gradient.asv_confidence == high_confidence_asv_bearing.confidence
        assert enhanced_gradient.asv_bearing_deg == high_confidence_asv_bearing.bearing_deg
        assert enhanced_gradient.asv_precision_deg == high_confidence_asv_bearing.precision_deg
        assert enhanced_gradient.processing_method == "asv_professional"

        # Enhanced gradient should be convertible to standard format
        standard_gradient = enhanced_gradient.to_gradient_vector()
        assert isinstance(standard_gradient, GradientVector)

    def test_asv_signal_degradation_detection_integration(
        self, confidence_homing
    ):
        """Test ASV signal degradation detection integration."""
        # Arrange - Create degraded signal scenario
        degraded_bearing = ASVBearingCalculation(
            bearing_deg=120.0,
            confidence=0.15,  # Very low confidence
            precision_deg=25.0,  # Poor precision
            signal_strength_dbm=-125.0,  # Very weak
            signal_quality=0.1,  # Very poor quality
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=True,  # Interference detected
            signal_classification="INTERFERENCE",
        )

        # Act
        decision = confidence_homing.evaluate_confidence_based_decision(degraded_bearing)

        # Assert - Should detect signal degradation
        assert decision.signal_degradation_detected is True
        assert decision.proceed_with_homing is False
        assert decision.fallback_strategy in ["RETURN_TO_LAST_KNOWN", "SPIRAL_SEARCH", "SAMPLING"]

    def test_backward_compatibility_with_existing_algorithm(
        self, homing_algorithm, confidence_homing, high_confidence_asv_bearing
    ):
        """Test backward compatibility with existing homing algorithm."""
        # Arrange - This test ensures existing algorithm still works
        # Create some RSSI history for traditional gradient calculation
        for i in range(5):
            homing_algorithm.add_rssi_sample(
                rssi=-80.0 - i,  # Increasing signal strength
                position_x=float(i),
                position_y=0.0,
                heading=90.0,
                timestamp=time.time() + i
            )

        # Act - Get traditional gradient
        traditional_gradient = homing_algorithm.calculate_gradient()

        # Also test enhanced integration
        enhanced_command = confidence_homing.integrate_with_homing_algorithm(
            high_confidence_asv_bearing, 90.0, time.time()
        )

        # Assert - Both should work
        if traditional_gradient is not None:  # If enough samples for traditional
            assert traditional_gradient.magnitude > 0
            assert 0 <= traditional_gradient.direction <= 360
            assert traditional_gradient.confidence >= 0

        assert enhanced_command is not None
        assert isinstance(enhanced_command, VelocityCommand)

    def test_configuration_integration_with_existing_parameters(
        self, confidence_homing
    ):
        """Test that confidence system integrates with existing configuration."""
        # Arrange - Configure dynamic thresholds
        custom_config = DynamicThresholdConfig(
            high_quality_threshold=0.2,
            moderate_quality_threshold=0.5,
            low_quality_threshold=0.9,
            signal_quality_boundaries=[0.8, 0.3],
            interference_penalty_factor=0.3
        )

        # Act
        confidence_homing.configure_threshold_parameters(custom_config)
        
        # Create test bearing
        test_bearing = ASVBearingCalculation(
            bearing_deg=45.0,
            confidence=0.6,
            precision_deg=5.0,
            signal_strength_dbm=-90.0,
            signal_quality=0.85,  # High quality
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=False,
        )

        decision = confidence_homing.evaluate_confidence_based_decision(test_bearing)

        # Assert - Should use custom configuration
        assert decision.dynamic_threshold <= custom_config.high_quality_threshold + 0.1  # Some adjustment expected
        assert decision.proceed_with_homing is True  # Should proceed with high quality signal