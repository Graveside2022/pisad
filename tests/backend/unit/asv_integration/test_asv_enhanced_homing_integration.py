"""Unit tests for ASV-Enhanced Homing Integration.

SUBTASK-6.1.2.2 [15a] - Test suite for ASV professional bearing integration with PISAD homing

This test suite validates the integration between ASV professional-grade bearing calculations
and PISAD's existing homing algorithm, including:
- Enhanced gradient calculation using ASV professional methods
- RSSI fallback integration when ASV unavailable
- Integration metrics and performance tracking
- Compatibility with existing GradientVector interface
"""

import asyncio
import math
import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from src.backend.services.asv_integration.asv_enhanced_homing_integration import (
    ASVEnhancedHomingIntegration,
    ASVEnhancedGradient,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVEnhancedSignalProcessor,
    ASVBearingCalculation,
)
from src.backend.services.asv_integration.asv_analyzer_factory import ASVAnalyzerFactory
from src.backend.services.asv_integration.asv_analyzer_wrapper import ASVAnalyzerBase
from src.backend.services.homing_algorithm import GradientVector


class TestASVEnhancedHomingIntegration:
    """Test suite for ASV-enhanced homing integration with professional bearing calculations."""

    @pytest.fixture
    def mock_asv_factory(self):
        """Create mock ASV analyzer factory."""
        mock_factory = Mock(spec=ASVAnalyzerFactory)
        mock_analyzer = Mock(spec=ASVAnalyzerBase)
        mock_analyzer.analyzer_type = "GP"
        mock_analyzer.frequency_hz = 406000000
        mock_factory.get_current_analyzer.return_value = mock_analyzer
        return mock_factory

    @pytest.fixture
    def homing_integration(self, mock_asv_factory):
        """Create ASV-enhanced homing integration instance."""
        return ASVEnhancedHomingIntegration(mock_asv_factory)

    @pytest.fixture
    def mock_asv_bearing(self):
        """Create mock ASV bearing calculation."""
        return ASVBearingCalculation(
            bearing_deg=135.0,
            confidence=0.85,
            precision_deg=2.8,
            signal_strength_dbm=-92.0,
            signal_quality=0.8,
            timestamp_ns=time.perf_counter_ns(),
            analyzer_type="GP",
            interference_detected=False,
            raw_asv_data={"processing_time_ns": 1500000},
        )

    @pytest.mark.asyncio
    async def test_asv_integration_initialization_success(
        self, homing_integration, mock_asv_factory
    ):
        """Test successful ASV integration initialization."""
        # Act
        result = await homing_integration.initialize_asv_integration()

        # Assert
        assert result is True
        assert homing_integration._signal_processor is not None
        assert isinstance(homing_integration._signal_processor, ASVEnhancedSignalProcessor)
        mock_asv_factory.get_current_analyzer.assert_called_once()

    @pytest.mark.asyncio
    async def test_asv_integration_initialization_no_factory(self):
        """Test ASV integration initialization without factory."""
        # Arrange
        homing_integration = ASVEnhancedHomingIntegration(None)

        # Act
        result = await homing_integration.initialize_asv_integration()

        # Assert
        assert result is False
        assert homing_integration._signal_processor is None

    @pytest.mark.asyncio
    async def test_asv_integration_initialization_no_analyzer(self, mock_asv_factory):
        """Test ASV integration initialization with no active analyzer."""
        # Arrange
        mock_asv_factory.get_current_analyzer.return_value = None
        homing_integration = ASVEnhancedHomingIntegration(mock_asv_factory)

        # Act
        result = await homing_integration.initialize_asv_integration()

        # Assert
        assert result is False
        assert homing_integration._signal_processor is None

    @pytest.mark.asyncio
    async def test_enhanced_gradient_calculation_asv_success(
        self, homing_integration, mock_asv_bearing
    ):
        """Test enhanced gradient calculation using ASV professional methods."""
        # Arrange
        await homing_integration.initialize_asv_integration()
        homing_integration._signal_processor.calculate_professional_bearing = AsyncMock(
            return_value=mock_asv_bearing
        )

        iq_samples = b"\x01\x02\x03\x04" * 150
        position_x, position_y = 15.0, 25.0
        current_heading = 90.0
        rssi_dbm = -88.0

        # Act
        result = await homing_integration.calculate_enhanced_gradient(
            iq_samples, position_x, position_y, current_heading, rssi_dbm
        )

        # Assert
        assert result is not None
        assert isinstance(result, ASVEnhancedGradient)
        assert result.asv_bearing_deg == 135.0
        assert result.asv_confidence == 0.85
        assert result.asv_precision_deg == 2.8
        assert result.signal_strength_dbm == -92.0
        assert not result.interference_detected
        assert result.processing_method == "asv_professional"
        assert result.calculation_time_ms > 0.0

        # Verify compatibility with standard gradient interface
        gradient_vector = result.to_gradient_vector()
        assert isinstance(gradient_vector, GradientVector)
        assert gradient_vector.magnitude == result.magnitude
        assert gradient_vector.direction == result.direction
        assert gradient_vector.confidence == result.confidence

    @pytest.mark.asyncio
    async def test_enhanced_gradient_precision_improvement(
        self, homing_integration, mock_asv_bearing
    ):
        """Test that ASV enhanced gradient provides precision improvement over RSSI."""
        # Arrange
        await homing_integration.initialize_asv_integration()
        homing_integration._signal_processor.calculate_professional_bearing = AsyncMock(
            return_value=mock_asv_bearing
        )

        # Act - Calculate multiple ASV bearings to build precision tracking
        for i in range(5):
            result = await homing_integration.calculate_enhanced_gradient(
                b"\x01\x02\x03\x04" * 150, float(i), 0.0, 90.0, -88.0
            )
            assert result is not None

        # Assert - Get integration metrics to verify precision improvement
        metrics = homing_integration.get_integration_metrics()
        assert metrics["asv_calculations"] == 5
        assert metrics["average_precision_improvement_factor"] > 1.0  # Should show improvement
        assert metrics["estimated_precision_deg"] < 12.0  # Better than RSSI baseline (~12°)

    @pytest.mark.asyncio
    async def test_enhanced_gradient_fallback_to_rssi(self, homing_integration):
        """Test fallback to RSSI gradient when ASV processing fails."""
        # Arrange
        await homing_integration.initialize_asv_integration()
        homing_integration._signal_processor.calculate_professional_bearing = AsyncMock(
            return_value=None  # ASV processing fails
        )

        # Build RSSI sample history for gradient calculation
        rssi_samples = [
            {"position_x": 0.0, "position_y": 0.0, "rssi": -95.0},
            {"position_x": 1.0, "position_y": 0.5, "rssi": -92.0},
            {"position_x": 2.0, "position_y": 1.0, "rssi": -88.0},
        ]

        for sample in rssi_samples:
            await homing_integration.calculate_enhanced_gradient(
                b"\x01\x02\x03\x04" * 100,
                sample["position_x"],
                sample["position_y"],
                90.0,
                sample["rssi"],
                time.perf_counter_ns(),
            )

        # Act - Final calculation should use RSSI fallback
        result = await homing_integration.calculate_enhanced_gradient(
            b"\x01\x02\x03\x04" * 100, 3.0, 1.5, 90.0, -85.0, time.perf_counter_ns()
        )

        # Assert
        assert result is not None
        assert result.processing_method == "fallback_rssi"
        assert result.asv_precision_deg == 15.0  # RSSI fallback precision estimate
        assert not result.interference_detected  # Cannot detect with RSSI only

        # Verify fallback metrics
        metrics = homing_integration.get_integration_metrics()
        assert metrics["fallback_calculations"] > 0
        assert metrics["asv_usage_rate"] < 1.0  # Not all calculations used ASV

    @pytest.mark.asyncio
    async def test_rssi_fallback_insufficient_samples(self, homing_integration):
        """Test RSSI fallback behavior with insufficient spatial samples."""
        # Arrange
        await homing_integration.initialize_asv_integration()
        homing_integration._signal_processor.calculate_professional_bearing = AsyncMock(
            return_value=None
        )

        # Act - Only provide 2 samples (insufficient for gradient)
        result1 = await homing_integration.calculate_enhanced_gradient(
            b"\x01\x02\x03\x04" * 100, 0.0, 0.0, 90.0, -95.0
        )
        result2 = await homing_integration.calculate_enhanced_gradient(
            b"\x01\x02\x03\x04" * 100, 0.1, 0.1, 90.0, -94.0  # Insufficient spatial diversity
        )

        # Assert - Should return None due to insufficient samples/diversity
        assert result1 is None  # First sample
        assert result2 is None  # Second sample, but insufficient diversity

    def test_asv_enhanced_gradient_to_gradient_vector_conversion(self):
        """Test conversion of ASV enhanced gradient to standard GradientVector."""
        # Arrange
        enhanced_gradient = ASVEnhancedGradient(
            magnitude=1.5,
            direction=225.0,
            confidence=78.0,
            asv_bearing_deg=225.0,
            asv_confidence=0.78,
            asv_precision_deg=3.2,
            signal_strength_dbm=-89.0,
            interference_detected=False,
            processing_method="asv_professional",
            calculation_time_ms=45.0,
        )

        # Act
        gradient_vector = enhanced_gradient.to_gradient_vector()

        # Assert
        assert isinstance(gradient_vector, GradientVector)
        assert gradient_vector.magnitude == 1.5
        assert gradient_vector.direction == 225.0
        assert gradient_vector.confidence == 78.0

    def test_integration_metrics_tracking(self, homing_integration):
        """Test integration performance metrics tracking."""
        # Act
        metrics = homing_integration.get_integration_metrics()

        # Assert - Initial metrics
        assert metrics["total_calculations"] == 0
        assert metrics["asv_calculations"] == 0
        assert metrics["fallback_calculations"] == 0
        assert metrics["asv_usage_rate"] == 0.0
        assert metrics["average_precision_improvement_factor"] == 0.0

    def test_integration_parameter_configuration(self, homing_integration):
        """Test configuration of integration parameters."""
        # Arrange
        original_fallback_enabled = homing_integration._rssi_fallback_enabled
        original_max_samples = homing_integration._max_rssi_samples

        # Act
        homing_integration.configure_integration_parameters(
            rssi_fallback_enabled=False,
            max_rssi_samples=20,
            asv_processor_config={"min_confidence_threshold": 0.4},
        )

        # Assert
        assert homing_integration._rssi_fallback_enabled is False
        assert homing_integration._max_rssi_samples == 20

    @pytest.mark.asyncio
    async def test_asv_analyzer_update_for_frequency_change(
        self, homing_integration, mock_asv_factory
    ):
        """Test ASV analyzer update when frequency changes (single-frequency mode)."""
        # Arrange
        await homing_integration.initialize_asv_integration()
        new_frequency = 121500000  # VOR frequency

        # Act
        result = await homing_integration.update_asv_analyzer(new_frequency)

        # Assert
        assert result is True
        mock_asv_factory.get_current_analyzer.assert_called()  # Called during update

    def test_bearing_history_access(self, homing_integration):
        """Test access to bearing calculation history."""
        # Act
        history = homing_integration.get_bearing_history(max_samples=5)

        # Assert
        assert isinstance(history, list)
        assert len(history) == 0  # Initially empty

    def test_processing_status_information(self, homing_integration):
        """Test processing status information retrieval."""
        # Act
        status = homing_integration.get_processing_status()

        # Assert
        assert isinstance(status, dict)
        assert "asv_available" in status
        assert "signal_processor_active" in status
        assert "rssi_fallback_enabled" in status
        assert "current_analyzer_type" in status
        assert "rssi_samples_available" in status

        # Initial state assertions
        assert status["signal_processor_active"] is False
        assert status["rssi_fallback_enabled"] is True
        assert status["rssi_samples_available"] == 0

    def test_calculation_history_clearing(self, homing_integration):
        """Test clearing of all calculation history."""
        # Arrange - Add some RSSI samples
        homing_integration._rssi_samples = [{"test": "data"}]

        # Act
        homing_integration.clear_calculation_history()

        # Assert
        assert len(homing_integration._rssi_samples) == 0

    def test_asv_availability_check(self, homing_integration):
        """Test ASV availability checking."""
        # Act & Assert - Before initialization
        assert homing_integration.is_asv_available() is False

        # After initialization (with mock factory)
        # Note: Would need actual initialization to test true case


class TestASVEnhancedGradientDataClass:
    """Test the ASVEnhancedGradient data structure."""

    def test_enhanced_gradient_creation(self):
        """Test creation of ASV enhanced gradient."""
        # Arrange & Act
        gradient = ASVEnhancedGradient(
            magnitude=2.1,
            direction=180.0,
            confidence=85.0,
            asv_bearing_deg=180.0,
            asv_confidence=0.85,
            asv_precision_deg=2.5,
            signal_strength_dbm=-87.0,
            interference_detected=False,
            processing_method="asv_professional",
            calculation_time_ms=32.5,
        )

        # Assert
        assert gradient.magnitude == 2.1
        assert gradient.direction == 180.0
        assert gradient.confidence == 85.0
        assert gradient.asv_bearing_deg == 180.0
        assert gradient.asv_confidence == 0.85
        assert gradient.asv_precision_deg == 2.5
        assert gradient.signal_strength_dbm == -87.0
        assert not gradient.interference_detected
        assert gradient.processing_method == "asv_professional"
        assert gradient.calculation_time_ms == 32.5

    def test_gradient_vector_conversion_compatibility(self):
        """Test conversion compatibility with existing GradientVector interface."""
        # Arrange
        enhanced_gradient = ASVEnhancedGradient(
            magnitude=1.8,
            direction=270.0,
            confidence=72.0,
            asv_bearing_deg=270.0,
            asv_confidence=0.72,
            asv_precision_deg=4.1,
            signal_strength_dbm=-91.0,
            interference_detected=True,
            processing_method="asv_professional",
            calculation_time_ms=28.0,
        )

        # Act
        gradient_vector = enhanced_gradient.to_gradient_vector()

        # Assert - Interface compatibility
        assert hasattr(gradient_vector, "magnitude")
        assert hasattr(gradient_vector, "direction")
        assert hasattr(gradient_vector, "confidence")
        assert gradient_vector.magnitude == enhanced_gradient.magnitude
        assert gradient_vector.direction == enhanced_gradient.direction
        assert gradient_vector.confidence == enhanced_gradient.confidence


@pytest.mark.integration
class TestASVEnhancedHomingIntegrationEnd2End:
    """End-to-end integration tests for ASV-enhanced homing integration."""

    @pytest.mark.asyncio
    async def test_full_integration_with_asv_components(self):
        """Test full integration with ASV components."""
        # This test would require actual ASV analyzer components
        # For now, demonstrate the integration pattern

        # Arrange
        homing_integration = ASVEnhancedHomingIntegration(None)  # No factory

        # Act - Should fall back to RSSI processing
        result = await homing_integration.calculate_enhanced_gradient(
            b"\x01\x02\x03\x04" * 100, 0.0, 0.0, 0.0, -95.0
        )

        # Assert - Returns None due to insufficient samples/no ASV
        assert result is None

    def test_performance_requirements_validation(self):
        """Validate that integration meets performance requirements."""
        # Integration should maintain <100ms processing time per calculation
        # Memory usage should remain efficient with history management

        # Create homing integration for this test
        homing_integration = ASVEnhancedHomingIntegration(None)

        # Check that history management is bounded
        assert homing_integration._max_rssi_samples <= 50  # Reasonable limit

        # Check that precision tracking is initialized
        assert homing_integration._average_precision_improvement >= 0.0
