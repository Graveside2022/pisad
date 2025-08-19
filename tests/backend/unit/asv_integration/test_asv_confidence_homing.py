"""Unit tests for ASV Confidence-Based Homing Integration.

SUBTASK-6.1.2.2 [15c] - Test suite for confidence-based signal quality assessment

This test suite validates the ASV confidence-based decision making integration including:
- Dynamic threshold adjustment based on ASV signal quality
- Confidence-based fallback strategies for course correction
- Integration with existing homing algorithm decision tree
- Preservation of all safety authority mechanisms
"""

import time
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
    ConfidenceBasedDecision,
    DynamicThresholdConfig,
)
from src.backend.services.asv_integration.asv_enhanced_homing_integration import (
    ASVEnhancedGradient,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVBearingCalculation,
    ASVEnhancedSignalProcessor,
)
from src.backend.services.homing_algorithm import (
    GradientVector,
    HomingAlgorithm,
)


@dataclass
class MockASVSignalQuality:
    """Mock ASV signal quality data for testing."""

    confidence: float
    signal_strength_dbm: float
    interference_detected: bool
    signal_quality: float


class TestASVConfidenceBasedHomingIntegration:
    """Test ASV confidence-based homing integration."""

    @pytest.fixture
    def mock_asv_processor(self):
        """Mock ASV enhanced signal processor."""
        processor = Mock(spec=ASVEnhancedSignalProcessor)
        return processor

    @pytest.fixture
    def mock_homing_algorithm(self):
        """Mock homing algorithm."""
        algorithm = Mock(spec=HomingAlgorithm)
        return algorithm

    @pytest.fixture
    def high_confidence_bearing(self):
        """High confidence bearing calculation from ASV."""
        return ASVBearingCalculation(
            bearing_deg=90.0,
            confidence=0.85,  # High confidence
            precision_deg=2.5,
            signal_strength_dbm=-75.0,
            signal_quality=0.9,  # High quality
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=False,
            signal_classification="CONTINUOUS",
        )

    @pytest.fixture
    def moderate_confidence_bearing(self):
        """Moderate confidence bearing calculation from ASV."""
        return ASVBearingCalculation(
            bearing_deg=120.0,
            confidence=0.45,  # Moderate confidence
            precision_deg=8.0,
            signal_strength_dbm=-95.0,
            signal_quality=0.6,  # Moderate quality
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=True,
            signal_classification="FM_CHIRP_WEAK",
        )

    @pytest.fixture
    def low_confidence_bearing(self):
        """Low confidence bearing calculation from ASV."""
        return ASVBearingCalculation(
            bearing_deg=45.0,
            confidence=0.15,  # Low confidence
            precision_deg=15.0,
            signal_strength_dbm=-115.0,
            signal_quality=0.2,  # Low quality
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=True,
            signal_classification="INTERFERENCE",
        )

    def test_dynamic_threshold_adjustment_high_quality(
        self, mock_asv_processor, mock_homing_algorithm, high_confidence_bearing
    ):
        """Test dynamic threshold adjustment for high quality signals."""
        # Arrange
        confidence_homing = ASVConfidenceBasedHoming(
            asv_processor=mock_asv_processor, homing_algorithm=mock_homing_algorithm
        )

        # Act
        decision = confidence_homing.evaluate_confidence_based_decision(high_confidence_bearing)

        # Assert - High quality signals should lower threshold requirements
        assert decision.proceed_with_homing is True
        assert decision.dynamic_threshold < 0.5  # Lowered from default
        assert decision.fallback_strategy == "NONE"
        assert decision.confidence_assessment == "HIGH"

    def test_dynamic_threshold_adjustment_moderate_quality(
        self, mock_asv_processor, mock_homing_algorithm, moderate_confidence_bearing
    ):
        """Test dynamic threshold adjustment for moderate quality signals."""
        # Arrange
        confidence_homing = ASVConfidenceBasedHoming(
            asv_processor=mock_asv_processor, homing_algorithm=mock_homing_algorithm
        )

        # Act
        decision = confidence_homing.evaluate_confidence_based_decision(moderate_confidence_bearing)

        # Assert - Moderate quality should use standard or slightly elevated thresholds
        assert decision.proceed_with_homing is False  # Below adjusted threshold
        assert decision.dynamic_threshold >= 0.5  # Standard or elevated
        assert decision.fallback_strategy in ["SAMPLING", "SPIRAL_SEARCH"]
        assert decision.confidence_assessment == "MODERATE"

    def test_confidence_based_fallback_strategies(
        self, mock_asv_processor, mock_homing_algorithm, low_confidence_bearing
    ):
        """Test confidence-based fallback strategies for weak signals."""
        # Arrange
        confidence_homing = ASVConfidenceBasedHoming(
            asv_processor=mock_asv_processor, homing_algorithm=mock_homing_algorithm
        )

        # Act
        decision = confidence_homing.evaluate_confidence_based_decision(low_confidence_bearing)

        # Assert - Low confidence should trigger fallback strategies
        assert decision.proceed_with_homing is False
        assert decision.fallback_strategy in ["RETURN_TO_LAST_KNOWN", "SPIRAL_SEARCH", "SAMPLING"]
        assert decision.confidence_assessment == "CRITICAL"  # Very low signal becomes CRITICAL
        assert decision.signal_degradation_detected is True

    def test_integration_with_existing_homing_decision_tree(
        self, mock_asv_processor, mock_homing_algorithm, high_confidence_bearing
    ):
        """Test integration with existing homing algorithm decision tree."""
        # Arrange
        confidence_homing = ASVConfidenceBasedHoming(
            asv_processor=mock_asv_processor, homing_algorithm=mock_homing_algorithm
        )

        # Mock existing homing algorithm gradient
        mock_gradient = GradientVector(magnitude=0.8, direction=90.0, confidence=85.0)

        # Act
        enhanced_gradient = confidence_homing.enhance_gradient_with_asv_confidence(
            original_gradient=mock_gradient, asv_bearing=high_confidence_bearing
        )

        # Assert - Should enhance but preserve original gradient interface
        assert isinstance(enhanced_gradient, ASVEnhancedGradient)
        assert enhanced_gradient.magnitude > 0
        assert enhanced_gradient.asv_confidence == high_confidence_bearing.confidence
        assert enhanced_gradient.processing_method == "asv_professional"

    def test_performance_requirement_decision_latency(
        self, mock_asv_processor, mock_homing_algorithm, moderate_confidence_bearing
    ):
        """Test that confidence-based decisions meet <100ms latency requirement."""
        # Arrange
        confidence_homing = ASVConfidenceBasedHoming(
            asv_processor=mock_asv_processor, homing_algorithm=mock_homing_algorithm
        )

        # Act - Measure decision time
        start_time = time.perf_counter()
        decision = confidence_homing.evaluate_confidence_based_decision(moderate_confidence_bearing)
        end_time = time.perf_counter()

        decision_latency_ms = (end_time - start_time) * 1000

        # Assert - Must meet performance requirement
        assert decision_latency_ms < 100.0  # <100ms requirement
        assert isinstance(decision, ConfidenceBasedDecision)

    def test_safety_authority_preservation(
        self, mock_asv_processor, mock_homing_algorithm, high_confidence_bearing
    ):
        """Test that all safety authority mechanisms are preserved."""
        # Arrange
        confidence_homing = ASVConfidenceBasedHoming(
            asv_processor=mock_asv_processor, homing_algorithm=mock_homing_algorithm
        )

        # Mock safety override scenario
        confidence_homing.set_safety_override(True, "Test safety override")

        # Act
        decision = confidence_homing.evaluate_confidence_based_decision(high_confidence_bearing)

        # Assert - Safety override should prevent homing regardless of confidence
        assert decision.proceed_with_homing is False
        assert decision.fallback_strategy == "SAFETY_OVERRIDE"
        assert decision.safety_override_reason is not None

    @pytest.mark.asyncio
    async def test_real_time_confidence_assessment_integration(
        self, mock_asv_processor, mock_homing_algorithm
    ):
        """Test real-time confidence assessment integration with signal processing."""
        # This test will use actual ASV signal processing integration
        # to verify authentic behavior - no mocking of ASV components

        # Arrange - This will be implemented after basic integration is working
        # For now, this is a placeholder for the authentic test requirement
        pass

    def test_configuration_parameters_for_dynamic_thresholds(self):
        """Test configuration of dynamic threshold parameters."""
        # Arrange
        config = DynamicThresholdConfig(
            high_quality_threshold=0.3,  # Lower threshold for high quality
            moderate_quality_threshold=0.6,  # Standard threshold
            low_quality_threshold=0.8,  # Higher threshold for low quality
            signal_quality_boundaries=[0.7, 0.4],  # High/Moderate/Low boundaries
            interference_penalty_factor=0.2,
        )

        # Act & Assert - Configuration should be applied correctly
        assert config.high_quality_threshold < config.moderate_quality_threshold
        assert config.moderate_quality_threshold < config.low_quality_threshold
        assert len(config.signal_quality_boundaries) == 2
        assert config.interference_penalty_factor > 0
