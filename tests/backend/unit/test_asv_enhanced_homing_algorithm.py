"""Test suite for ASV-enhanced homing algorithm gradient calculation replacement.

TASK-6.1.16a: Replace basic homing_algorithm.py gradient calculation with ASV enhanced algorithms

This test suite validates the integration of ASV professional bearing calculation
methods into the existing homing algorithm, ensuring:
- Enhanced precision (±2° vs ±10°)
- Backward compatibility with existing GradientVector interface
- Real ASV integration (no mocks)
"""

import math

import pytest

from src.backend.services.asv_integration.asv_enhanced_homing_integration import (
    ASVEnhancedHomingIntegration,
)
from src.backend.services.homing_algorithm import (
    GradientVector,
    HomingAlgorithm,
    RSSISample,
)


class TestASVEnhancedGradientCalculation:
    """Test ASV-enhanced gradient calculation integration."""

    @pytest.fixture
    def homing_algorithm(self):
        """Create homing algorithm instance for testing."""
        return HomingAlgorithm()

    @pytest.fixture
    def asv_integration(self):
        """Create ASV integration instance for testing."""
        # Real ASV integration - no mocks
        return ASVEnhancedHomingIntegration()

    def test_asv_enhanced_gradient_calculation_replaces_numpy(
        self, homing_algorithm, asv_integration
    ):
        """
        TDD RED PHASE: Test that ASV enhanced calculation replaces numpy-based gradient calculation.

        [16a-4] Replace numpy-based gradient calculation with ASV ASVEnhancedHomingIntegration calls
        This test should FAIL initially since the integration is not implemented yet.
        """
        # Add RSSI samples to create spatial diversity for gradient calculation
        samples = [
            RSSISample(rssi=-75.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0),
            RSSISample(rssi=-70.0, position_x=10.0, position_y=0.0, heading=90.0, timestamp=2.0),
            RSSISample(rssi=-65.0, position_x=20.0, position_y=0.0, heading=180.0, timestamp=3.0),
        ]

        # Add samples with IQ data to trigger ASV enhancement
        for sample in samples:
            homing_algorithm.add_rssi_sample(
                sample.rssi,
                sample.position_x,
                sample.position_y,
                sample.heading,
                sample.timestamp,
                iq_samples=b"\x00\x01" * 512,  # Provide mock IQ samples to trigger ASV path
            )

        # This should call ASV enhanced calculation instead of numpy least squares
        gradient = homing_algorithm.calculate_gradient()

        # Verify ASV integration was used (this will fail until implementation)
        assert gradient is not None, "ASV-enhanced gradient calculation should return valid result"
        assert hasattr(homing_algorithm, "_asv_integration"), "Should have ASV integration instance"
        assert (
            homing_algorithm._asv_integration is not None
        ), "ASV integration should be initialized"

    def test_asv_bearing_precision_improvement(self, homing_algorithm, asv_integration):
        """
        TDD RED PHASE: Test bearing precision improvement from ±10° to ±2°.

        [16a-6] Test bearing precision improvements with ±2° accuracy target validation
        """
        # Create test scenario with known bearing (45 degrees)
        true_bearing = 45.0

        # Add samples that should produce the known bearing
        samples = self._generate_bearing_test_samples(true_bearing)
        # Add samples with IQ data to trigger ASV enhancement
        for sample in samples:
            homing_algorithm.add_rssi_sample(
                sample.rssi,
                sample.position_x,
                sample.position_y,
                sample.heading,
                sample.timestamp,
                iq_samples=b"\x00\x01" * 512,  # Provide mock IQ samples
            )

        gradient = homing_algorithm.calculate_gradient()

        # Verify enhanced precision - should be within ±2° of true bearing
        assert gradient is not None, "Should calculate gradient for precision test"
        bearing_error = abs(gradient.direction - true_bearing)
        if bearing_error > 180:
            bearing_error = 360 - bearing_error  # Handle wraparound

        assert (
            bearing_error <= 2.0
        ), f"ASV bearing error {bearing_error:.1f}° should be ≤2° (target precision)"

        # Verify confidence is high for ASV calculation
        assert (
            gradient.confidence > 90.0
        ), f"ASV confidence {gradient.confidence:.1f}% should be >90%"

    def test_gradient_vector_compatibility_preserved(self, homing_algorithm):
        """
        TDD RED PHASE: Test that GradientVector interface compatibility is maintained.

        [16a-2] Design ASV enhanced gradient calculation interface that maintains GradientVector compatibility
        [16a-7] Ensure backward compatibility with all existing VelocityCommand generation patterns
        """
        # Add test samples
        samples = [
            RSSISample(rssi=-80.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0),
            RSSISample(rssi=-75.0, position_x=5.0, position_y=5.0, heading=45.0, timestamp=2.0),
            RSSISample(rssi=-70.0, position_x=10.0, position_y=10.0, heading=90.0, timestamp=3.0),
        ]

        for sample in samples:
            homing_algorithm.add_rssi_sample(
                sample.rssi, sample.position_x, sample.position_y, sample.heading, sample.timestamp
            )

        gradient = homing_algorithm.calculate_gradient()

        # Verify GradientVector structure is preserved
        assert isinstance(gradient, GradientVector), "Should return GradientVector instance"
        assert hasattr(gradient, "magnitude"), "Should have magnitude field"
        assert hasattr(gradient, "direction"), "Should have direction field"
        assert hasattr(gradient, "confidence"), "Should have confidence field"

        # Verify field types are correct
        assert isinstance(gradient.magnitude, float), "Magnitude should be float"
        assert isinstance(gradient.direction, float), "Direction should be float"
        assert isinstance(gradient.confidence, float), "Confidence should be float"

        # Verify direction is normalized 0-360
        assert (
            0.0 <= gradient.direction <= 360.0
        ), f"Direction {gradient.direction}° should be 0-360°"

    def test_asv_integration_initialization(self, homing_algorithm):
        """
        TDD RED PHASE: Test that ASV integration is properly initialized.

        [16a-3] Implement ASV professional bearing data integration into existing gradient structures
        """
        # Verify ASV integration components are available
        assert hasattr(homing_algorithm, "_asv_integration"), "Should initialize ASV integration"

        # Verify ASV integration can provide enhanced calculations
        asv_integration = homing_algorithm._asv_integration
        assert asv_integration is not None, "ASV integration should be initialized"

        # Test that ASV integration has required methods
        assert hasattr(
            asv_integration, "calculate_enhanced_gradient"
        ), "Should have enhanced calculation method"

    def test_method_signatures_preserved(self, homing_algorithm):
        """
        TDD RED PHASE: Test that existing method signatures are preserved.

        [16a-5] Preserve existing compute_optimal_heading() and scale_velocity_by_gradient() method signatures
        """
        # Create test gradient
        test_gradient = GradientVector(magnitude=1.5, direction=135.0, confidence=85.0)

        # Test compute_optimal_heading signature
        optimal_heading = homing_algorithm.compute_optimal_heading(test_gradient)
        assert isinstance(optimal_heading, float), "compute_optimal_heading should return float"
        assert 0.0 <= optimal_heading <= 360.0, "Optimal heading should be normalized"

        # Test scale_velocity_by_gradient signature
        scaled_velocity = homing_algorithm.scale_velocity_by_gradient(test_gradient)
        assert isinstance(scaled_velocity, float), "scale_velocity_by_gradient should return float"
        assert scaled_velocity > 0.0, "Scaled velocity should be positive"

    def _generate_bearing_test_samples(self, true_bearing_deg: float) -> list[RSSISample]:
        """Generate test samples that should produce a specific bearing."""
        samples = []
        bearing_rad = math.radians(true_bearing_deg)

        # Generate samples along the bearing line with increasing RSSI
        for i in range(5):
            distance = i * 5.0  # 0, 5, 10, 15, 20 meters
            x = distance * math.cos(bearing_rad)
            y = distance * math.sin(bearing_rad)

            # RSSI increases as we move toward the beacon
            rssi = -90.0 + (i * 5.0)  # -90, -85, -80, -75, -70 dBm

            sample = RSSISample(
                rssi=rssi,
                position_x=x,
                position_y=y,
                heading=true_bearing_deg,
                timestamp=float(i + 1),
            )
            samples.append(sample)

        return samples

    def test_performance_requirement_maintained(self, homing_algorithm):
        """
        TDD RED PHASE: Test that <100ms processing latency is maintained with ASV integration.

        Technical Requirement: Processing latency remains <100ms per computation cycle
        """
        import time

        # Add sufficient samples for gradient calculation
        samples = [
            RSSISample(rssi=-80.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0),
            RSSISample(rssi=-75.0, position_x=10.0, position_y=5.0, heading=30.0, timestamp=2.0),
            RSSISample(rssi=-70.0, position_x=20.0, position_y=10.0, heading=60.0, timestamp=3.0),
            RSSISample(rssi=-65.0, position_x=30.0, position_y=15.0, heading=90.0, timestamp=4.0),
        ]

        for sample in samples:
            homing_algorithm.add_rssi_sample(
                sample.rssi, sample.position_x, sample.position_y, sample.heading, sample.timestamp
            )

        # Measure calculation time
        start_time = time.perf_counter()
        gradient = homing_algorithm.calculate_gradient()
        end_time = time.perf_counter()

        calculation_time_ms = (end_time - start_time) * 1000.0

        assert gradient is not None, "Should calculate gradient for performance test"
        assert (
            calculation_time_ms < 100.0
        ), f"ASV calculation took {calculation_time_ms:.2f}ms, should be <100ms"
